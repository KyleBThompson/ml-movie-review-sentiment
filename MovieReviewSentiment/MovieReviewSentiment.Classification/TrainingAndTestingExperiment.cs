using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MovieReviewSentiment.Classification
{
  public class TrainingAndTestingExperiment
  {
    private readonly IList<Document> _documents = new List<Document>();
    private readonly NaiveBayes _cl;
    private readonly string _inputFilesPath;

    public TrainingAndTestingExperiment(string inputFilePath = null)
    {
      Func<string, IList<string>> getFeatures = delegate(string s)
      {
        var docParser = new DocumentParser();
        docParser.AddItem(s);
        return docParser.GetFeatures();
      };
      if (inputFilePath != null)
      {
        _inputFilesPath = inputFilePath + "/";
      }
      _cl = new NaiveBayes(getFeatures);
    }

    public AccuracyResult TestForAccuracy()
    {
      LoadDocuments();
      LoadDocumentsFromKagglePositiveNegativeOnly();
      var recs = _documents.Count;
      var labels = GetLabels();
      var accuracy = CalculateAccuracy(labels);
      var result = new AccuracyResult
      {
        Accuracy = accuracy,
        TrainingDatasetSize = recs,
        Classifier = _cl
      };
      return result;
    }

    public double TestForAccuracyUsingKaggleTestData()
    {
      LoadDocumentsFromKaggle();
      var labels = GetLabelsForKaggle();
      return CalculateAccuracy(labels);
    }

    private double CalculateAccuracy(string[] labels)
    {
      var splitMap = new Dictionary<string, double>();
      var labelDocuments = new Dictionary<string, List<Document>>();

      var skipped = 0;
      var correct = 0.0;
      var incorrect = 0.0;
      var totalSplit = Math.Floor(0.85*_documents.Count);
      var trainedCount = 0;

      foreach (var label in labels)
      {
        var documents = _documents.Where(x => x.Label == label).OrderBy(y => Guid.NewGuid()).ToList();
        labelDocuments[label] = documents;
        var length = documents.Count;
        var split = Math.Floor(0.85*length);
        splitMap[label] = split;

        for (var i = 0; i < split; i++)
        {
          _cl.Train(documents[i].Item, documents[i].Label);
          if (i%1000 == 0)
          {
            if (i != 0)
            {
              trainedCount += 1000;
            }
            var trainingPct = Math.Round(trainedCount*100/totalSplit);
            Console.WriteLine("Training progress: " + trainingPct + "%");
          }
        }
      }

      foreach (var label in labels)
      {
        var documents = labelDocuments[label];
        var length = documents.Count;
        var split = splitMap[label];

        for (var i = split; i < length; i++)
        {
          var result = _cl.Classify(documents[(int) i].Item);
          if (result.Probability < 0.75)
          {
            skipped++;
          }
          else if (result.Label == label)
          {
            correct++;
          }
          else
          {
            incorrect++;
          }
        }
      }

      var resultsPct = Math.Round(10000*correct/(correct + incorrect))/100;
      return resultsPct;
    }

    private static string[] GetLabels()
    {
      var labels = new[] {"positive", "negative"};
      return labels;
    }

    private void LoadDocuments()
    {
      using (var reader = new StreamReader(_inputFilesPath + "files/positive.txt"))
      {
        string line;
        while ((line = reader.ReadLine()) != null)
        {
          _documents.Add(new Document {Label = "positive", Item = line});
        }
      }
      using (var reader = new StreamReader(_inputFilesPath + "files/negative.txt"))
      {
        string line;
        while ((line = reader.ReadLine()) != null)
        {
          _documents.Add(new Document {Label = "negative", Item = line});
        }
      }
    }

    private static string[] GetLabelsForKaggle()
    {
      var labels = new[] { "0", "1", "2", "3", "5" };
      return labels;
    }

    private void LoadDocumentsFromKaggle()
    {
      using (var reader = new StreamReader(_inputFilesPath + "files/train.tsv"))
      {
        reader.ReadLine(); // skip header
        string line;
        while ((line = reader.ReadLine()) != null)
        {
          var review = line.Split('\t');
          var phrase = review[2];
          var sentiment = review[3];
          _documents.Add(new Document { Label = sentiment, Item = phrase });
        }
      }
    }

    private void LoadDocumentsFromKagglePositiveNegativeOnly()
    {
      using (var reader = new StreamReader(_inputFilesPath + "files/train.tsv"))
      {
        reader.ReadLine(); // skip header
        string line;
        while ((line = reader.ReadLine()) != null)
        {
          var review = line.Split('\t');
          var phrase = review[2];
          var sentiment = review[3];
          switch (sentiment)
          {
            case "0":
              sentiment = "negative";
              break;
            case "1":
              sentiment = "negative";
              break;
            case "2":
              sentiment = "skip";
              break;
            case "3":
              sentiment = "positive";
              break;
            case "4":
              sentiment = "positive";
              break;
          }

          if (sentiment == "skip") continue;
          _documents.Add(new Document { Label = sentiment, Item = phrase });
        }
      }
    }

    private class Document
    {
      public string Label { get; set; }
      public string Item { get; set; }
    }
  }

  public class AccuracyResult
  {
    public double Accuracy { get; set; }
    public int TrainingDatasetSize { get; set; }
    public Classifier Classifier { get; set; }
  }

}
