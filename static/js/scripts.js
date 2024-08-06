$(document).ready(function() {
    $("#contributionForm").submit(function(event) {
        event.preventDefault();

        var formData = {
            amountPaid: $("#amountPaid").val(),
            region: $("#region").val(),
            occupation: $("#occupation").val()
        };

        $.ajax({
            type: "POST",
            url: "http://127.0.0.1:8000/predict",
            data: JSON.stringify(formData),
            contentType: "application/json",
            dataType: "json",
            success: function(response) {
                $("#result").html("Predicted Contribution Amount: " + response.predicted_contribution_amount.toFixed(2));
            },
            error: function(error) {
                console.log("Error: ", error);
                $("#result").html("Error predicting contribution amount.");
            }
        });
    });
});
