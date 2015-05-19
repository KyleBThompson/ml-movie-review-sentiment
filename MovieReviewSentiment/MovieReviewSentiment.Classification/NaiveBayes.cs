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
      var scores = CalculateProbabilityForEachLabel(text);
      var bestScore = scores.Aggregate((l, r) => l.Value > r.Value ? l : r);

      return new Classification
      {
        Label = bestScore.Key,
        Probability = bestScore.Value
      };
    }

    public Dictionary<string, double> CalculateProbabilityForEachLabel(string item)
    {
      var labels = LabelItemCount.Select(x => x.Key).ToList();
      var words = GetFeatures(item);
      var scores = new Dictionary<string, double>();

      labels.ForEach(x => scores[x] = CalculateDocumentProbabilityForLabel(words, x));

      return scores;
    }

    private double CalculateDocumentProbabilityForLabel(IEnumerable<string> words, string label)
    {
      var logSum = 0.0;
      foreach (var word in words)
      {
        var nbrTimesWordAppearsInADocument = FeatureCount(word);
        if (nbrTimesWordAppearsInADocument == 0) { continue; }

        var probGivenWordFitsLabel = CalculateProbabilityThatGivenWordFitsLabel(word, label);

        probGivenWordFitsLabel = AccountForRareWords(probGivenWordFitsLabel, nbrTimesWordAppearsInADocument);

        // combine probabilities, prevent floating point loss of precision
        logSum += (Math.Log(1 - probGivenWordFitsLabel) - Math.Log(probGivenWordFitsLabel));

        Log(string.Format("{0}icity of {1}: {2}", label, word, probGivenWordFitsLabel));
      }
      return 1 / (1 + Math.Exp(logSum));
    }

    private double CalculateProbabilityThatGivenWordFitsLabel(string word, string label)
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

      return probGivenWordFitsLabel;
    }

    private static double AccountForRareWords(double probGivenWordFitsLabel, int nbrTimesWordAppearsInADocument)
    {
      const int weight = 10;
      const double valueToAdjustTowards = 0.5;
      probGivenWordFitsLabel = ((weight * valueToAdjustTowards) 
                               + (nbrTimesWordAppearsInADocument * probGivenWordFitsLabel))
                               / (weight + nbrTimesWordAppearsInADocument);
      return probGivenWordFitsLabel;
    }

  }

  public class Classification
  {
    public string Label { get; set; }
    public double Probability { get; set; }
  }
}