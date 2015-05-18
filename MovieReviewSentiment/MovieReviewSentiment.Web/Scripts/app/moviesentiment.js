﻿


$('#guess').click(function () {
  var review = { text: $('#review').val() };
  $.ajax({
    type: "POST",
    data: JSON.stringify(review),
    dataType: 'json',
    url: "api/moviereview/guess",
    contentType: "application/json",
    success: function (data) {
      if (data.confidence === 0) {
        $("#training-result").empty().append(data.label);
      } else {
        $("#sentiment").empty().append(data.label);
        $("#confidence").append("Confidence Level: " + data.confidence + "%");
        $("#result").show();
      }
    }
  });

});

$('#train').click(function () {
  $("#training-result").empty().append("Training the classifer, this could take a moment...");
  $.ajax({
    type: "POST",
    dataType: 'json',
    url: "api/moviereview/train",
    contentType: "application/json",
    success: function (data) {
      $("#training-result").empty().append("Classifier has been trained with " + Number(data.trainingDatasetSize).toLocaleString('en') + " records.");
      $("#training-result").append("<br/>Accuracy from self test = " + data.accuracy + "%");
    }
  });

});