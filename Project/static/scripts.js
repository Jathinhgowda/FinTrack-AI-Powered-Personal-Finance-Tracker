// JavaScript for real-time interactivity

// Handle form validation for transactions
document.addEventListener('DOMContentLoaded', () => {
    const transactionForm = document.querySelector('form[action="/transactions"]');
    if (transactionForm) {
        transactionForm.addEventListener('submit', (e) => {
            const description = transactionForm.querySelector('input[name="description"]').value.trim();
            const amount = parseFloat(transactionForm.querySelector('input[name="amount"]').value);
            const category = transactionForm.querySelector('select[name="category"]').value;

            if (!description || isNaN(amount) || !category) {
                e.preventDefault();
                alert("Please fill in all the fields correctly!");
            }
        });
    }
});

// Load AI insights
const fetchAIInsights = async () => {
    const insightsContainer = document.getElementById('ai-insights-container');
    if (!insightsContainer) return;

    try {
        // Example payload - Replace with dynamic data
        const payload = {
            features: [500, 1] // Replace with actual transaction features (amount, category)
        };

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            const data = await response.json();
            insightsContainer.textContent = `Predicted Spending Trend: ${data.trend_prediction}`;
        } else {
            throw new Error("Failed to fetch AI insights");
        }
    } catch (error) {
        insightsContainer.textContent = `Error: ${error.message}`;
    }
};

// Automatically load AI insights when the page loads
document.addEventListener('DOMContentLoaded', fetchAIInsights);

// Confirmation prompt for transaction deletion
const deleteButtons = document.querySelectorAll('form[action^="/transactions/delete"] button');
deleteButtons.forEach(button => {
    button.addEventListener('click', (e) => {
        if (!confirm("Are you sure you want to delete this transaction?")) {
            e.preventDefault();
        }
    });
});
