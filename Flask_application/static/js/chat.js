const form = document.getElementById('chat-form');
const input = document.getElementById('user-input');
const windowDiv = document.getElementById('chat-window');
const genCharts = document.getElementById('generated-charts');

function appendMessage(who, text) {
    const div = document.createElement('div');
    div.className = who;
    div.innerText = text;
    windowDiv.appendChild(div);
    windowDiv.scrollTop = windowDiv.scrollHeight;
}

form.addEventListener('submit', async e => {
    e.preventDefault();
    const msg = input.value.trim();
    if (!msg) return;
    appendMessage('user', msg);
    input.value = '';
    const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg })
    });
    const data = await res.json();
    appendMessage('assistant', data.reply);
    if (data.chart_url) {
        const img = document.createElement('img');
        img.src = data.chart_url;
        genCharts.appendChild(img);
    }
});
