// EduPulse Dashboard Logic

// State
let appHistory = JSON.parse(localStorage.getItem('edupulse_history') || '[]');

// DOM Elements
const views = {
    analysis: document.getElementById('view-analysis'),
    report: document.getElementById('view-report'),
    history: document.getElementById('view-history')
};

const els = {
    input: document.getElementById('reviewInput'),
    charCount: document.getElementById('charCount'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    btnText: document.querySelector('.btn-text'),
    btnLoader: document.getElementById('btnLoader'),
    
    resultState: document.getElementById('resultState'),
    resultContent: document.getElementById('resultContent'),
    
    // Result Fields
    starRating: document.getElementById('starRating'),
    sentimentBadge: document.getElementById('sentimentBadge'),
    confScore: document.getElementById('confScore'),
    confFill: document.getElementById('confFill'),
    infTime: document.getElementById('infTime'),
    wordCount: document.getElementById('wordCount'),
    
    historyTable: document.getElementById('historyTableBody'),
    pageTitle: document.getElementById('pageTitle'),
    wandbFrame: document.getElementById('wandbFrame')
};

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    updateCharCount();
    renderHistory();
});

// Navigation
function switchTab(tabName) {
    // Update Active Link
    document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
    event.currentTarget.classList.add('active');
    
    // Show View
    Object.values(views).forEach(view => view.classList.remove('active'));
    views[tabName].classList.add('active');
    
    // Update Title
    const titles = {
        'analysis': 'Live Analysis',
        'report': 'Model Performance Report',
        'history': 'Analysis History'
    };
    els.pageTitle.textContent = titles[tabName];
}

// Input Handling
els.input.addEventListener('input', updateCharCount);

function updateCharCount() {
    els.charCount.textContent = els.input.value.length;
}

function clearInput() {
    els.input.value = '';
    updateCharCount();
    showPlaceholder();
}

function showPlaceholder() {
    els.resultState.classList.remove('hidden');
    els.resultState.style.display = 'flex';
    els.resultContent.classList.add('hidden');
}

function showResult(data) {
    els.resultState.classList.add('hidden');
    els.resultState.style.display = 'none';
    els.resultContent.classList.remove('hidden');
    
    els.starRating.textContent = getStarString(data.star_rating);
    
    // Sentiment Badge
    els.sentimentBadge.textContent = data.sentiment;
    els.sentimentBadge.className = `sentiment-badge ${data.sentiment.toLowerCase()}`;
    
    // Confidence
    // If confidence is a string "High", map it? No, app.py returns string now?
    // Wait app.py returns "High" etc. but user wants score percentage?
    // app.py returns 'score' as float (0.8).
    
    const confPct = Math.round(data.score * 100);
    els.confScore.textContent = `${confPct}%`;
    els.confFill.style.width = `${confPct}%`;
    
    // Stats
    els.infTime.textContent = `${data.inference_time_ms}ms`;
    els.wordCount.textContent = data.word_count;
}

function getStarString(rating) {
    const rounded = Math.round(rating * 2) / 2;
    const full = Math.floor(rounded);
    const half = (rounded % 1) !== 0;
    return "★".repeat(full) + (half ? "½" : "") + "☆".repeat(5 - Math.ceil(rounded));
}


// API Call
async function analyzeSentiment() {
    const text = els.input.value.trim();
    if (!text) return;
    
    setLoading(true);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        
        const data = await response.json();
        
        showResult(data);
        addToHistory(text, data);
        
    } catch (e) {
        console.error(e);
        alert("Analysis failed. Check console.");
    } finally {
        setLoading(false);
    }
}

function setLoading(isLoading) {
    if (isLoading) {
        els.btnText.classList.add('hidden');
        els.btnLoader.classList.remove('hidden');
        els.btnLoader.style.display = 'block';
    } else {
        els.btnText.classList.remove('hidden');
        els.btnLoader.classList.add('hidden');
        els.btnLoader.style.display = 'none';
        els.btnLoader.style.removeProperty('display');
    }
}

// History
function addToHistory(text, result) {
    const item = {
        time: new Date().toLocaleTimeString(),
        text: text,
        rating: result.star_rating,
        sentiment: result.sentiment,
        conf: Math.round(result.score * 100) + '%'
    };
    
    appHistory.unshift(item);
    if (appHistory.length > 20) appHistory.pop();
    localStorage.setItem('edupulse_history', JSON.stringify(appHistory));
    
    renderHistory();
}

function renderHistory() {
    els.historyTable.innerHTML = appHistory.map(item => {
        const rating = (item.rating !== undefined && item.rating !== null) ? Number(item.rating).toFixed(1) : 'N/A';
        const sentiment = item.sentiment || 'Unknown';
        const conf = item.conf || '-';
        return `
        <tr>
            <td style="color:var(--text-muted)">${item.time}</td>
            <td class="truncate" title="${item.text}">${item.text.substring(0, 40)}...</td>
            <td>⭐ ${rating}</td>
            <td><span class="sentiment-badge ${sentiment.toLowerCase()}" style="font-size:10px; padding:2px 8px">${sentiment}</span></td>
            <td>${conf}</td>
        </tr>
    `}).join('');
}

function clearHistory() {
    appHistory = [];
    localStorage.removeItem('edupulse_history');
    renderHistory();
}

function reloadIframe() {
    els.wandbFrame.src = els.wandbFrame.src;
}