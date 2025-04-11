document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector('form');
    const resultArea = document.getElementById('result-area');

    form.addEventListener('submit', () => {
        if (resultArea) {
            resultArea.style.display = 'flex';
            resultArea.innerHTML = `
                <div id="status-message" style="text-align:center;">
                    <div class="inline-spinner"></div>
                    <p style="margin-top: 10px;">Génération en cours...</p>
                </div>
            `;
        }
    });
});
