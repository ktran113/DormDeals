const API_URL = "";

const token = localStorage.getItem('token');
//if (!token) window.location.href = 'login.html';

// Populate sidebar user info
const userName  = localStorage.getItem('userName')  || 'User';
const userEmail = localStorage.getItem('userEmail') || '';
document.getElementById('user-name').textContent  = userName;
document.getElementById('user-email').textContent = userEmail;
document.getElementById('user-avatar').textContent = userName.charAt(0).toUpperCase();

// Logout
document.getElementById('logout-btn').addEventListener('click', function () {
    ['token','userId','userEmail','userName'].forEach(k => localStorage.removeItem(k));
    window.location.href = 'index.html';
});

// Upload elements
const uploadZone    = document.getElementById('upload-zone');
const uploadPreview = document.getElementById('upload-preview');
const imageInput    = document.getElementById('image-input');
const previewImg    = document.getElementById('preview-img');
const browseBtn     = document.getElementById('browse-btn');
const clearBtn      = document.getElementById('clear-btn');
const searchBtn     = document.getElementById('search-btn');

browseBtn.addEventListener('click', () => imageInput.click());

imageInput.addEventListener('change', function () {
    if (this.files[0]) showPreview(this.files[0]);
});

uploadZone.addEventListener('dragover', function (e) {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', function (e) {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) showPreview(file);
});

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
        previewImg.src = e.target.result;
        uploadZone.style.display = 'none';
        uploadPreview.style.display = 'flex';
    };
    reader.readAsDataURL(file);
}

clearBtn.addEventListener('click', function () {
    imageInput.value = '';
    previewImg.src = '';
    uploadZone.style.display = 'flex';
    uploadPreview.style.display = 'none';
    hideSkeleton();
    document.getElementById('results-panel').style.display = 'none';
    document.getElementById('tips-panel').style.display = 'block';
});

// Skeleton helpers
function showSkeleton() {
    const panel = document.getElementById('results-panel');
    const tips  = document.getElementById('tips-panel');
    document.getElementById('skeleton-grid').style.display = 'grid';
    document.getElementById('results-grid').style.display  = 'none';
    panel.style.display = 'block';
    tips.style.display  = 'none';
}

function hideSkeleton() {
    document.getElementById('skeleton-grid').style.display = 'none';
    document.getElementById('results-grid').style.display  = 'grid';
}

searchBtn.addEventListener('click', async function () {
    const file = imageInput.files[0];
    if (!file) return;

    searchBtn.textContent = 'Searching...';
    searchBtn.disabled = true;
    showSkeleton();

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/search`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}` },
            body: formData
        });

        if (response.status === 401) { window.location.href = 'login.html'; return; }

        const data = await response.json();
        if (response.ok) {
            showResults(data.results || []);
        } else {
            hideSkeleton();
            alert(data.detail || 'Search failed. Please try again.');
        }
    } catch (error) {
        hideSkeleton();
        alert('Search is not yet available — the backend feature is coming soon!');
    } finally {
        searchBtn.textContent = 'Find matching deals';
        searchBtn.disabled = false;
    }
});

function showResults(results) {
    const grid = document.getElementById('results-grid');
    grid.innerHTML = '';

    if (results.length === 0) {
        grid.innerHTML = '<p class="no-results">No similar items found. Try a different photo.</p>';
    } else {
        results.forEach(item => {
            const card = document.createElement('div');
            card.className = 'result-card card card-lift';
            card.innerHTML = `
                ${item.image_url ? `<img src="${item.image_url}" alt="${item.title || 'Item'}" onerror="this.style.display='none'">` : ''}
                <div class="result-info">
                    <h4>${item.title || 'Similar Item'}</h4>
                    <div class="result-price">${item.price ? '$' + item.price : 'Price unlisted'}</div>
                    <div class="result-source">${item.source || 'Local Marketplace'}</div>
                    <span class="matched-pill">
                        <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
                        Matched by photo
                    </span>
                    ${item.url ? `<br><a href="${item.url}" target="_blank" rel="noopener" class="btn btn-primary" style="font-size:0.8rem;padding:6px 12px;margin-top:8px;display:inline-block;">View listing</a>` : ''}
                </div>
            `;
            grid.appendChild(card);
        });
    }
    hideSkeleton();
}
