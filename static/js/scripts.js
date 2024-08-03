document.getElementById('predictForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const demographics = document.getElementById('demographics').value;
    const location = document.getElementById('location').value;
    const income_group = document.getElementById('income_group').value;
    const work_type = document.getElementById('work_type').value;

    const data = {
        demographics: demographics,
        location: location,
        income_group: income_group,
        work_type: work_type
    };

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        if (data.error) {
            resultDiv.textContent = `Error: ${data.error}`;
        } else {
            resultDiv.textContent = `Predicted Contribution: ${data.prediction}`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
