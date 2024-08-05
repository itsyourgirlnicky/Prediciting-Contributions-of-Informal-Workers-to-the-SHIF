$(document).ready(function() {
    $('#contributionForm').on('submit', function(event) {
        event.preventDefault();

        var formData = {
            amountPaid: $('#amountPaid').val(),
            region: $('#region').val(),
            occupation: $('#occupation').val()
        };

        $.ajax({
            url: '/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(formData),
            success: function(response) {
                if (response.prediction) {
                    $('#result').html('<h4>Predicted Contribution: ' + response.prediction + '</h4>');
                } else if (response.error) {
                    $('#result').html('<h4>Error: ' + response.error + '</h4>');
                }
            },
            error: function(xhr, status, error) {
                console.log(xhr.responseText);
                $('#result').html('<h4>An unexpected error occurred: ' + xhr.responseText + '</h4>');
            }
        });
    });
});

