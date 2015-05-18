using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MovieReviewSentiment.Classification
{
  public class Loader
  {
    public static void ClassifyWithKaggleFiles(NaiveBayes cl, Dictionary<string, string> testReviews = null)
    {
      Console.WriteLine("{0} {1} {2}", Environment.NewLine, "Kaggle Data...", Environment.NewLine);

      TrainWithKaggleTextFiles(cl);

      if (testReviews == null)
      {
        return;
      }
      foreach (var testReview in testReviews)
      {
        Console.WriteLine();
        var result = cl.Classify(testReview.Value);
        Console.WriteLine(testReview.Value);
        var sentiment = ConvertKaggleSentiment(result.Label);
        Console.WriteLine("{0} {1}", sentiment, Math.Round((result.Probability * 100), 2));
      }
    }

    public static void ClassifyWithTestFiles(NaiveBayes cl, Dictionary<string, string> testReviews = null)
    {
      cl.SetDebug(false);
      TrainWithTextFiles(cl);

      if (testReviews == null)
      {
        return;
      }
      foreach (var testReview in testReviews)
      {
        var result = cl.Classify(testReview.Value);
        Console.WriteLine("{0} :: {1}", testReview.Key, testReview.Value);
        Console.WriteLine("{0} {1}", result.Label, Math.Round((result.Probability * 100), 2));
        Console.WriteLine("");
      }
    }

    public static void TrainWithTextFiles(Classifier cl)
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

    public static void ClassifyAndTestWithKaggleFiles(NaiveBayes cl)
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

    public static void TrainWithKaggleTextFiles(Classifier cl)
    {
      using (var reader = new StreamReader("files/train.tsv"))
      {
        reader.ReadLine(); // skip header
        var counter = 1;
        string line;
        while ((line = reader.ReadLine()) != null)
        {
          if (counter % 10000 == 0)
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

    public static string ConvertKaggleSentiment(string sentiment)
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
