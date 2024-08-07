$(document).ready(function() {
    $('#contributionForm').on('submit', function(e) {
        e.preventDefault();

        var amountPaid = $('#amountPaid').val();
        var region = $('#region').val();
        var occupation = $('#occupation').val();

        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                amountPaid: parseFloat(amountPaid),
                region: region,
                occupation: occupation
            }),
            success: function(response) {
                var resultHtml = `
                    <div class="card">
                        <div class="card-header">
                            Predicted Contribution Amount
                        </div>
                        <div class="card-body">
                            <h5 class="card-title">Your Predicted Contribution</h5>
                            <p class="card-text">Based on your inputs, the predicted contribution amount is: <strong>Ksh ${response.predicted_contribution.toFixed(2)}</strong></p>
                        </div>
                    </div>
                `;
                $('#result').html(resultHtml);
            },
            error: function(error) {
                $('#result').html('<div class="alert alert-danger">An error occurred while predicting the contribution amount.</div>');
            }
        });
    });
});
