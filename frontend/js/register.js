const API_URL = "";
const form = document.getElementById('register-form');
const errorDiv   = document.getElementById('error-message');
const successDiv = document.getElementById('success-message');
const loadingDiv = document.getElementById('loading');

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    successDiv.style.display = 'none';
}

function showSuccess(message) {
    successDiv.textContent = message;
    successDiv.style.display = 'block';
    errorDiv.style.display = 'none';
}

function hideMessages() {
    errorDiv.style.display = 'none';
    successDiv.style.display = 'none';
}
function showLoading() {
    loadingDiv.style.display = 'block';
    form.style.display = 'none';
}

function hideLoading() {
    loadingDiv.style.display = 'none';
    form.style.display = 'block';
}

//listen for form submission
form.addEventListener('submit', async function(event) {

    event.preventDefault();

    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    console.log('User wants to register:', { name, email });

    const userData = {
        name: name,
        email: email,
        password: password
    };

    //spinner
    showLoading();
    try {
        const response = await fetch(`${API_URL}/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },

            body: JSON.stringify(userData)
        });
        const data = await response.json();

        // successful request
        if (response.ok) {
            console.log('Registration successful!', data);

            // save token and user info
            localStorage.setItem('token', data.access_token);
            localStorage.setItem('userId', data.user_id);
            localStorage.setItem('userEmail', data.email);
            localStorage.setItem('userName', data.name);
            showSuccess('Account created successfully! Redirecting...');
            setTimeout(function() {
                window.location.href = 'dashboard.html';
            }, 2000);

        } else {
            showError(data.detail || 'Registration failed. Please try again.');
            hideLoading();
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Cannot connect to server. Make sure backend is running!');
        hideLoading();
    }
});