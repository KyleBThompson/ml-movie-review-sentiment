using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MovieReviewSentiment.Classification;

namespace MovieReviewSentiment
{
  class Program
  {
    static void Main(string[] args)
    {
      Console.WriteLine(new TrainingAndTestingExperiment().TestForAccuracy().Accuracy);
   //   Console.WriteLine(new TrainingAndTestingExperiment().TestForAccuracyUsingKaggleTestData());
      Console.WriteLine("Accuracy test complete -----------------------------------------------");
      Console.WriteLine("");
     // Console.ReadLine();
     // return;
      
      var testReviews = new Dictionary<string, string>();
      testReviews["avengers_meh"] = "More does not necessarily equal better; here, more is just ... meh.";
      testReviews["avengers_shorterbigger"] = "Age of Ultron is a minute shorter than its predecessor, but it's a bigger movie in nearly every other regard.";
      testReviews["avengers_heckofaparty"] = "The Avengers, we're told, are greater than the sum of their parts. Avengers: Age of Ultron is not. But it still boasts some pretty incredible parts. And one heck of a party.";
      testReviews["madmax_worththewait"] = "With Fury Road, director George Miller returns to the lawless, oil-deprived future of his seminal series for the first time in three decades. It was worth the wait.";
      testReviews["mallcop_forgettable"] = "Paul Blart: Mall Cop 2 may be forgettable, but it's difficult to truly dislike.";
      testReviews["exmachina_smartslick"] = "One of the smartest, slickest and most intriguing sci-fi thrillers in recent memory, with a standout performances from Isaac and Vikander and confident, minimalist direction from Garland.";
      testReviews["almostmarried_somewhatnegative"] = "There are a handful of truly funny moments, however there isn't quite enough fluency to the dialogue.";

      Func<string, IList<string>> getFeatures = delegate(string s)
      {
        var docParser = new DocumentParser();
        docParser.AddItem(s);
        return docParser.GetFeatures();
      };

      var cl = new NaiveBayes(getFeatures);

      // train with smaller dataset (positive and negative only), use testReviews as input
      Loader.ClassifyWithTestFiles(cl, testReviews);
      
      // train with kaggle data (big dataset), use testReviews as input
     // ClassifyWithKaggleFiles(cl, getFeatures, testReviews); 

      // trans and classifies kaggles input data and writes results to file (bin/debug/files/submission.tsv
      //ClassifyAndTestWithKaggleFiles(cl, getFeatures); 

      while (true)
      {
        Console.WriteLine("");
        Console.WriteLine("Enter Review:");
        string line = Console.ReadLine(); 
        if (line == "exit")
        {
          break;
        }
        var c = cl.Classify(line);
        Console.WriteLine("");
        var sentiment = Loader.ConvertKaggleSentiment(c.Label);
        Console.WriteLine("Sentiment: {0} Confidence: {1}", sentiment, Math.Round((c.Probability * 100), 2));
      }
    }



  }
}
