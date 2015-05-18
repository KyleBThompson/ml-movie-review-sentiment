using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;

namespace MovieReviewSentiment.Classification
{
  public class NaiveBayes : Classifier
  {
    public NaiveBayes()
    {
    }
    public NaiveBayes(Func<string, IList<string>> getFeatures) : base(getFeatures)
    {
    }

    public double DocumentProbability(string item, string label)
     {
      var features = GetFeatures(item);
      
      // Multiply the probabilities of all the features together
      double p = 1;
      foreach (var feature in features)
      {
        var wp = FeatureProbability(feature, label);
        Console.WriteLine("{0} {1} {2}", feature, label, wp);
        p*= wp;
        Console.WriteLine("running prob for {0} is {1}", label, p);
      }
      Console.WriteLine("doc prob for {0} is {1}", label, p);
      return p;
    }

    public Classification Classify(string text)
    {
      if (LabelItemCount.Count == 0)
      {
        throw new ConstraintException("Classifier has not been trained.");
      }
      var scores = Probability(text);
      double[] bestScore = { 0.0 };
      var bestLabel = string.Empty;
      foreach (var label in scores.Where(label => label.Value > bestScore[0]))
      {
        bestScore[0] = label.Value;
        bestLabel = label.Key;
      }

      return new Classification
      {
        Label = bestLabel,
        Probability = bestScore[0]
      };
    }

    public Dictionary<string, double> Probability(string item)
    {
      var labels = LabelItemCount.Select(x => x.Key).ToList();
      var words = GetFeatures(item);
      var scores = new Dictionary<string, double>();

      foreach (var label in labels)
      {
        var logSum = 0.0;
        foreach (var word in words)
        {
          var wordTotal = FeatureCount(word);
          if (wordTotal == 0)
          {
            continue;
          }
          var wordCountForLabel = FeatureCountForLabel(word, label);
          var docCountForLabel = DocumentCount(label);
          var wordProbForLabel = wordCountForLabel/docCountForLabel;
          var wordCountForOtherLabel = InverseFeatureCount(word, label);
          var docCountForOtherLabel = InverseDocumentCount(label);
          var wordProbOtherLabel = wordCountForOtherLabel/docCountForOtherLabel;

          var probGivenWorkFitLabel = wordProbForLabel/(wordProbForLabel + wordProbOtherLabel);

          probGivenWorkFitLabel = ((3 * 0.5) + (wordTotal * probGivenWorkFitLabel)) / (3 + wordTotal);

          if (probGivenWorkFitLabel.Equals(0.0))
          {
            probGivenWorkFitLabel = 0.00001;
          }
          else if (probGivenWorkFitLabel.Equals(1.0))
          {
            probGivenWorkFitLabel = 0.99999;
          }
          
          logSum += (Math.Log(1 - probGivenWorkFitLabel) - Math.Log(probGivenWorkFitLabel));
          Log(string.Format("{0}icity of {1}: {2}",label,word, probGivenWorkFitLabel));
        }
        scores[label] = 1/(1 + Math.Exp(logSum));
      }

      return scores;
    }

  }

  public class Classification
  {
    public string Label { get; set; }
    public double Probability { get; set; }
  }
}
