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
      Console.WriteLine(new TrainingAndTestingExperiment().TestForAccuracy());
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
      ClassifyWithTestFiles(cl, getFeatures, testReviews);
      
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
        var sentiment = ConvertKaggleSentiment(c.Label);
        Console.WriteLine("Sentiment: {0} Confidence: {1}", sentiment, Math.Round((c.Probability * 100), 2));
      }
    }


    private static void ClassifyWithKaggleFiles(NaiveBayes cl, Func<string, IList<string>> getFeatures, Dictionary<string, string> testReviews)
    {
      Console.WriteLine("{0} {1} {2}", Environment.NewLine, "Kaggle Data...", Environment.NewLine);

      TrainWithKaggleTextFiles(cl);

      foreach (var testReview in testReviews)
      {
        Console.WriteLine();
        var result = cl.Classify(testReview.Value);
        Console.WriteLine(testReview.Value);
        var sentiment = ConvertKaggleSentiment(result.Label);
        Console.WriteLine("{0} {1}", sentiment, Math.Round((result.Probability*100), 2));
      }
    }

    private static void ClassifyWithTestFiles(NaiveBayes cl, Func<string, IList<string>> getFeatures, Dictionary<string, string> testReviews)
    {
      cl.SetDebug(false);
      TrainWithTextFiles(cl);

      foreach (var testReview in testReviews)
      {
        var result = cl.Classify(testReview.Value);
        Console.WriteLine("{0} :: {1}", testReview.Key, testReview.Value);
        Console.WriteLine("{0} {1}", result.Label, Math.Round((result.Probability*100), 2));
        Console.WriteLine("");
      }
    }

    private static void TrainWithTextFiles(Classifier cl)
    {
      using (var reader = new StreamReader("files/positive.txt"))
      {
        string line;
        while ((line = reader.ReadLine()) != null)
        {
          cl.Train(line, "positive");
        }
      }

      using (var reader = new StreamReader("files/negative.txt"))
      {
        string line;
        while ((line = reader.ReadLine()) != null)
        {
          cl.Train(line, "negative");
        }
      }
      
    }

    private static void ClassifyAndTestWithKaggleFiles(NaiveBayes cl, Func<string, IList<string>> getFeatures)
    {
      const string submissionPath = "files/submission.tsv";
      if (File.Exists(submissionPath))
      {
        File.Delete(submissionPath);
      }

      TrainWithKaggleTextFiles(cl);

      Console.Write(Environment.NewLine);

      using (var writer = new StreamWriter(submissionPath))
      {
        var counter = 1;
        using (var reader = new StreamReader("files/test.tsv"))
        {
          reader.ReadLine(); //header
          string line;
          while ((line = reader.ReadLine()) != null)
          {
            if (counter % 1000 == 0)
            {
              Console.Write(counter + ", ");
            }
            var review = line.Split('\t');
            var phraseId = review[0];
            var phrase = review[2];
            var result = cl.Classify(phrase);
            var sentiment = ConvertKaggleSentiment(result.Label);
            writer.WriteLine(@"{0} {2}({3}) ""{1}""", phraseId, phrase, sentiment, result.Label);
            counter++;
          }
        }
      }

      Console.WriteLine(Environment.NewLine + "Classification Complete");
    }

    private static void TrainWithKaggleTextFiles(Classifier cl)
    {
      using (var reader = new StreamReader("files/train.tsv"))
      {
        reader.ReadLine(); // skip header
        var counter = 1;
        string line;
        while ((line = reader.ReadLine()) != null)
        {
          if (counter%10000 == 0)
          {
            Console.Write(counter + ", ");
          }

          var review = line.Split('\t');
          var phrase = review[2];
          var sentiment = review[3];

          cl.Train(phrase, sentiment);
          counter++;
        }
      }
      Console.WriteLine(Environment.NewLine + "Training Complete");
    }

    private static string ConvertKaggleSentiment(string sentiment)
    {
      switch (sentiment)
      {
        case "0":
          sentiment = "negative";
          break;
        case "1":
          sentiment = "somewhat negative";
          break;
        case "2":
          sentiment = "neutral";
          break;
        case "3":
          sentiment = "somewhat positive";
          break;
        case "4":
          sentiment = "positive";
          break;
      }
      return sentiment;
    }
  }
}
