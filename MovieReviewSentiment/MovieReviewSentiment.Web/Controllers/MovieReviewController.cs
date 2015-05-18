using System;
using System.Collections.Generic;
using System.Data;
using System.Net;
using System.Net.Http;
using System.Web;
using System.Web.Http;
using MovieReviewSentiment.Classification;

namespace MovieReviewSentiment.Web.Controllers
{
  public class MovieReviewController : ApiController
  {
    // POST api/MovieReview
    public HttpResponseMessage Guess([FromBody]MovieReview movieReview)
    {
      var result = new ClassificationResult();

      if (string.IsNullOrEmpty(movieReview.Text))
      {
        result.Confidence = 0.0;
        result.Label = "Do I look like a mind reader?";
        return Request.CreateResponse(HttpStatusCode.Created, result);
      }

      var cl = (NaiveBayes)HttpContext.Current.Cache["Classifier"];
      if (cl == null)
      {
        result.Confidence = 0.0;
        result.Label = "I have no clue. Did you forget to train me?";
        return Request.CreateResponse(HttpStatusCode.Created, result);
      }
      try
      {
        var c = cl.Classify(movieReview.Text);
        result.Label = UppercaseFirst(c.Label);
        result.Confidence = Math.Round((c.Probability * 100), 2);
      }
      catch (ConstraintException)
      {
        result.Confidence = 0.0;
        result.Label = "I have no clue. Did you forget to train me?";
      }

      return Request.CreateResponse(HttpStatusCode.Created, result);
    }

    // POST api/MovieReview/train
    [HttpPost]
    public HttpResponseMessage Train()
    {
      var inputFilesPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "bin");

      var result = new TrainingAndTestingExperiment(inputFilesPath).TestForAccuracy();
      HttpContext.Current.Cache["Classifier"] = result.Classifier;

      return Request.CreateResponse(HttpStatusCode.Created, result);
    }

    private static string UppercaseFirst(string s)
    {
      // Check for empty string.
      if (string.IsNullOrEmpty(s))
      {
        return string.Empty;
      }
      // Return char and concat substring.
      return char.ToUpper(s[0]) + s.Substring(1);
    }
  }

  public class MovieReview
  {
    public string Text { get; set; }
  }

  public class ClassificationResult
  {
    public string Label { get; set; }
    public double Confidence { get; set; }
  }
}