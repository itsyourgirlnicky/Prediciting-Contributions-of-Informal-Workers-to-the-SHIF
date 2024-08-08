$(document).ready(function() {
    $('#contributionForm').on('submit', function(event) {
        event.preventDefault();
        const amountPaid = $('#amountPaid').val();
        const region = $('#region').val();
        const occupation = $('#occupation').val();

        if (region === "" || occupation === "") {
            alert("Please select both region and occupation.");
            return;
        }

        $.ajax({
            url: '/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                amountPaid: parseFloat(amountPaid),
                region: region,
                occupation: occupation
            }),
            success: function(response) {
                $('#result').html(`<p>Predicted SHIF Contribution: ${response.predicted_contribution}</p>`);
            },
            error: function(error) {
                $('#result').html(`<p>Error: ${error.responseJSON.detail}</p>`);
            }
        });
    });
});
