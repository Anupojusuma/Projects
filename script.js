// Toggle Dark/Light Mode
const toggleBtn = document.getElementById('toggle-btn');
toggleBtn.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    document.body.classList.toggle('light-mode');
});

// Render Model Performance Chart
const ctx = document.getElementById('modelChart').getContext('2d');
const labels = Object.keys(scores);
const data = {
    labels: labels,
    datasets: [
        {
            label: 'AUC Score',
            data: labels.map(model => scores[model]['AUC']),
            backgroundColor: '#007bff',
            borderRadius: 8
        }
    ]
};

const config = {
    type: 'bar',
    data: data,
    options: {
        responsive: true,
        plugins: {
            legend: { display: false },
            title: { display: true, text: 'Model AUC Comparison' }
        }
    }
};

new Chart(ctx, config);

// Load Lottie Animation
var animation = bodymovin.loadAnimation({
    container: document.getElementById('animation'),
    renderer: 'svg',
    loop: true,
    autoplay: true,
    path: 'https://assets9.lottiefiles.com/packages/lf20_0yfsb3a1.json'  // Example 3D Animation
});
