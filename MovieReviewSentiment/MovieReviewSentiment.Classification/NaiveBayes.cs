using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;

namespace MovieReviewSentiment.Classification
{
  public class NaiveBayes : Classifier
  {
    public NaiveBayes(){}
    public NaiveBayes(Func<string, IList<string>> getFeatures) : base(getFeatures){}

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
        CalculateDocumentProbabilityForLabel(words, label, scores);
      }

      return scores;
    }

    private void CalculateDocumentProbabilityForLabel(IEnumerable<string> words, string label, IDictionary<string, double> scores)
    {
      var logSum = 0.0;
      foreach (var word in words)
      {
        var wordTotal = FeatureCount(word);
        if (wordTotal == 0)
        {
          continue;
        }
        var probGivenWordFitsLabel = CalculateProbabilityThatGivenWordFitsLabel(word, label, wordTotal);

        logSum += (Math.Log(1 - probGivenWordFitsLabel) - Math.Log(probGivenWordFitsLabel));
        Log(string.Format("{0}icity of {1}: {2}", label, word, probGivenWordFitsLabel));
      }
      scores[label] = 1/(1 + Math.Exp(logSum));
    }

    private double CalculateProbabilityThatGivenWordFitsLabel(string word, string label, int wordTotal)
    {
      // Given a specific label, the probability this word fits
      var wordCountForLabel = FeatureCountForLabel(word, label);
      var docCountForLabel = DocumentCount(label);
      var wordProbForLabel = wordCountForLabel/docCountForLabel;
      // Now do inverse for probability that ehword fits some other label
      var wordCountForOtherLabels = InverseFeatureCount(word, label);
      var docCountForOtherLabels = InverseDocumentCount(label);
      var wordProbOtherLabels = wordCountForOtherLabels/docCountForOtherLabels;

      // Bayes magic: Given this word, what is probability it should be classified with this label
      var probGivenWordFitsLabel = wordProbForLabel/(wordProbForLabel + wordProbOtherLabels);

      // account for rare words
      probGivenWordFitsLabel = ((3*0.5) + (wordTotal*probGivenWordFitsLabel))/(3 + wordTotal);

      return probGivenWordFitsLabel;
    }
  }

  public class Classification
  {
    public string Label { get; set; }
    public double Probability { get; set; }
  }
}