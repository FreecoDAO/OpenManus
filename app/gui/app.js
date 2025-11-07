// FreEco.AI GUI Application

const API_BASE = '/api';

// ===== STATE =====

let currentTab = 'chat';
let isDarkTheme = localStorage.getItem('theme') === 'dark';

// ===== INITIALIZATION =====

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initEventListeners();
    loadInitialData();
    checkHealth();
});

// ===== THEME =====

function initTheme() {
    if (isDarkTheme) {
        document.body.classList.add('dark-theme');
        document.getElementById('themeToggle').textContent = '☀️';
    }
}

document.getElementById('themeToggle').addEventListener('click', () => {
    isDarkTheme = !isDarkTheme;
    document.body.classList.toggle('dark-theme');
    document.getElementById('themeToggle').textContent = isDarkTheme ? '☀️' : '🌙';
    localStorage.setItem('theme', isDarkTheme ? 'dark' : 'light');
});

// ===== EVENT LISTENERS =====

function initEventListeners() {
    // Tab navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            switchTab(e.target.dataset.tab);
        });
    });

    // Chat
    document.getElementById('sendBtn').addEventListener('click', sendMessage);
    document.getElementById('chatInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // Tasks
    document.getElementById('newTaskBtn').addEventListener('click', () => {
        openModal('taskModal');
    });
    document.getElementById('taskForm').addEventListener('submit', createTask);

    // Knowledge Base
    document.getElementById('newDocBtn').addEventListener('click', () => {
        openModal('docModal');
    });
    document.getElementById('docForm').addEventListener('submit', addDocument);
    document.getElementById('searchInput').addEventListener('input', searchDocuments);

    // Settings
    document.getElementById('settingsForm').addEventListener('submit', updateSettings);
    document.getElementById('testBtn').addEventListener('click', testConnection);
    document.getElementById('temperature').addEventListener('input', (e) => {
        document.getElementById('tempValue').textContent = e.target.value;
    });

    // Modals
    document.querySelectorAll('.close-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            closeModal(e.target.dataset.modal);
        });
    });

    document.querySelectorAll('[data-modal]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            if (e.target.dataset.modal) {
                closeModal(e.target.dataset.modal);
            }
        });
    });
}

// ===== TAB SWITCHING =====

function switchTab(tab) {
    currentTab = tab;

    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tab) {
            btn.classList.add('active');
        }
    });

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${tab}-tab`).classList.add('active');

    // Load data for tab
    if (tab === 'tasks') {
        loadTasks();
    } else if (tab === 'knowledge') {
        loadDocuments();
    } else if (tab === 'settings') {
        loadSettings();
    }
}

// ===== CHAT =====

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();

    if (!message) return;

    // Add user message to UI
    addChatMessage('user', message);
    input.value = '';

    // Show loading
    showLoading(true);
    setStatus('Sending message...');

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        if (!response.ok) throw new Error('Failed to send message');

        const data = await response.json();
        addChatMessage('assistant', data.response);
        setStatus('');
    } catch (error) {
        addChatMessage('system', `Error: ${error.message}`);
        setStatus(`Error: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

function addChatMessage(role, text) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}`;
    messageEl.innerHTML = `<p>${escapeHtml(text)}</p>`;
    messagesDiv.appendChild(messageEl);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function setStatus(text) {
    document.getElementById('chatStatus').textContent = text;
}

// ===== TASKS =====

async function loadTasks() {
    try {
        const [tasksRes, statsRes] = await Promise.all([
            fetch(`${API_BASE}/tasks`),
            fetch(`${API_BASE}/tasks/stats`)
        ]);

        const tasks = await tasksRes.json();
        const stats = await statsRes.json();

        // Update stats
        document.getElementById('statTotal').textContent = stats.total;
        document.getElementById('statPending').textContent = stats.pending;
        document.getElementById('statInProgress').textContent = stats.in_progress;
        document.getElementById('statCompleted').textContent = stats.completed;

        // Render tasks
        const taskList = document.getElementById('taskList');
        taskList.innerHTML = '';

        if (tasks.length === 0) {
            taskList.innerHTML = '<p style="color: var(--text-secondary);">No tasks yet. Create one to get started!</p>';
            return;
        }

        tasks.forEach(task => {
            const taskEl = document.createElement('div');
            taskEl.className = 'task-item';
            taskEl.innerHTML = `
                <div class="task-item-content">
                    <div class="task-item-title">${escapeHtml(task.title)}</div>
                    <div class="task-item-meta">
                        Priority: <strong>${task.priority}</strong> |
                        Status: <strong>${task.status}</strong>
                    </div>
                </div>
                <div class="task-item-actions">
                    <button onclick="updateTaskStatus('${task.id}', 'in_progress')">Start</button>
                    <button onclick="updateTaskStatus('${task.id}', 'completed')">Complete</button>
                    <button onclick="deleteTask('${task.id}')">Delete</button>
                </div>
            `;
            taskList.appendChild(taskEl);
        });
    } catch (error) {
        console.error('Error loading tasks:', error);
    }
}

async function createTask(e) {
    e.preventDefault();

    const title = document.getElementById('taskTitle').value;
    const description = document.getElementById('taskDesc').value;
    const priority = document.getElementById('taskPriority').value;

    try {
        const response = await fetch(`${API_BASE}/tasks`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, description, priority })
        });

        if (!response.ok) throw new Error('Failed to create task');

        closeModal('taskModal');
        document.getElementById('taskForm').reset();
        loadTasks();
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function updateTaskStatus(taskId, status) {
    try {
        const response = await fetch(`${API_BASE}/tasks/${taskId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status })
        });

        if (!response.ok) throw new Error('Failed to update task');

        loadTasks();
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function deleteTask(taskId) {
    if (!confirm('Are you sure?')) return;

    try {
        const response = await fetch(`${API_BASE}/tasks/${taskId}`, {
            method: 'DELETE'
        });

        if (!response.ok) throw new Error('Failed to delete task');

        loadTasks();
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// ===== KNOWLEDGE BASE =====

async function loadDocuments() {
    try {
        const [docsRes, statsRes] = await Promise.all([
            fetch(`${API_BASE}/documents`),
            fetch(`${API_BASE}/documents/stats`)
        ]);

        const docs = await docsRes.json();
        const stats = await statsRes.json();

        // Update stats
        document.getElementById('statDocs').textContent = stats.total_documents;
        document.getElementById('statSize').textContent = stats.total_size_mb + ' MB';

        // Render documents
        renderDocuments(docs);
    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

function renderDocuments(docs) {
    const docList = document.getElementById('documentList');
    docList.innerHTML = '';

    if (docs.length === 0) {
        docList.innerHTML = '<p style="color: var(--text-secondary);">No documents yet. Add one to get started!</p>';
        return;
    }

    docs.forEach(doc => {
        const docEl = document.createElement('div');
        docEl.className = 'document-item';
        docEl.innerHTML = `
            <div class="document-item-content">
                <div class="document-item-title">${escapeHtml(doc.title)}</div>
                <div class="document-item-meta">
                    Category: <strong>${doc.category}</strong> |
                    Size: <strong>${(doc.size / 1024).toFixed(2)} KB</strong>
                </div>
            </div>
            <div class="document-item-actions">
                <button onclick="deleteDocument('${doc.id}')">Delete</button>
            </div>
        `;
        docList.appendChild(docEl);
    });
}

async function addDocument(e) {
    e.preventDefault();

    const title = document.getElementById('docTitle').value;
    const content = document.getElementById('docContent').value;
    const category = document.getElementById('docCategory').value;

    try {
        const response = await fetch(`${API_BASE}/documents`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, content, category })
        });

        if (!response.ok) throw new Error('Failed to add document');

        closeModal('docModal');
        document.getElementById('docForm').reset();
        loadDocuments();
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function searchDocuments(e) {
    const query = e.target.value.trim();

    if (!query) {
        loadDocuments();
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/documents/search?q=${encodeURIComponent(query)}`);
        const docs = await response.json();
        renderDocuments(docs);
    } catch (error) {
        console.error('Error searching:', error);
    }
}

async function deleteDocument(docId) {
    if (!confirm('Are you sure?')) return;

    try {
        const response = await fetch(`${API_BASE}/documents/${docId}`, {
            method: 'DELETE'
        });

        if (!response.ok) throw new Error('Failed to delete document');

        loadDocuments();
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// ===== SETTINGS =====

async function loadSettings() {
    try {
        const response = await fetch(`${API_BASE}/settings`);
        const settings = await response.json();

        document.getElementById('model').value = settings.minimax.model;
        document.getElementById('temperature').value = settings.minimax.temperature;
        document.getElementById('tempValue').textContent = settings.minimax.temperature;
        document.getElementById('maxTokens').value = settings.minimax.max_tokens;
    } catch (error) {
        console.error('Error loading settings:', error);
    }
}

async function updateSettings(e) {
    e.preventDefault();

    const apiKey = document.getElementById('apiKey').value;
    const model = document.getElementById('model').value;
    const temperature = parseFloat(document.getElementById('temperature').value);
    const maxTokens = parseInt(document.getElementById('maxTokens').value);

    try {
        const response = await fetch(`${API_BASE}/settings`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_key: apiKey, model, temperature, max_tokens: maxTokens })
        });

        if (!response.ok) throw new Error('Failed to update settings');

        alert('Settings updated successfully!');
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function testConnection() {
    showLoading(true);

    try {
        const response = await fetch(`${API_BASE}/settings/test-minimax`, {
            method: 'POST'
        });

        const data = await response.json();
        const statusEl = document.getElementById('connectionStatus');

        if (data.connected) {
            statusEl.innerHTML = '<span class="status-indicator connected"></span><span>Connected</span>';
        } else {
            statusEl.innerHTML = '<span class="status-indicator disconnected"></span><span>Connection failed</span>';
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// ===== MODALS =====

function openModal(modalId) {
    document.getElementById(modalId).classList.add('active');
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

// ===== UTILITIES =====

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (show) {
        overlay.classList.add('active');
    } else {
        overlay.classList.remove('active');
    }
}

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        console.log('Health check:', data);
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

async function loadInitialData() {
    await loadTasks();
    await loadDocuments();
    await loadSettings();
}
