using System;
using System.Collections.Generic;
using System.Linq;

namespace MovieReviewSentiment.Classification
{
  public class Classifier
  {
    protected readonly Func<string, IList<string>> GetFeatures;
    private readonly IDictionary<string, IDictionary<string, int>> _featureLabels;
    protected readonly IDictionary<string, double> LabelItemCount;
    protected readonly IDictionary<string, double> Thresholds;
    private bool _debug;

    public Classifier(){
    }

    public Classifier(Func<string, IList<string>> getFeatures)
    {
      GetFeatures = getFeatures;
      _featureLabels = 
        new Dictionary<string, IDictionary<string, int>>(StringComparer.OrdinalIgnoreCase);
      LabelItemCount = new Dictionary<string, double>();
      Thresholds = new Dictionary<string, double>();
    }
    
    public void Train(string item, string label)
    {
      var features = GetFeatures(item);
      foreach (var f in features)
      {
        IncrementFeatureLabelCount(label, f);
      }

      if (!LabelItemCount.ContainsKey(label))
      {
        LabelItemCount[label] = 0;
      }
      LabelItemCount[label]++;
    }

    public void IncrementFeatureLabelCount(string label, string feature)
    {
      if (_featureLabels.ContainsKey(feature))
      {
        var fc = _featureLabels[feature];
        if (!fc.ContainsKey(label))
        {
          fc[label] = 0;
        }
        fc[label]++;
      }
      else
      {
        _featureLabels[feature] = 
          new Dictionary<string, int>( StringComparer.OrdinalIgnoreCase){{label, 1}};
      }
    }

    public int FeatureCount(string feature)
    {
      return !_featureLabels.ContainsKey(feature)
        ? 0
        : _featureLabels[feature].Sum(labelCount => labelCount.Value);
    }

    public int FeatureCountForLabel(string feature, string label)
    {
      if (!_featureLabels.ContainsKey(feature))
      {
        return 0;
      }
      return !_featureLabels[feature].ContainsKey(label)
        ? 0
        : _featureLabels[feature][label];
    }

    public int InverseFeatureCount(string feature, string label)
    {
      if (!_featureLabels.ContainsKey(feature))
      {
        return 0;
      }
      var featureCat = _featureLabels[feature];
      return featureCat.Where(x => x.Key != label).Sum(x => x.Value);
    }

    public double DocumentCount(string label)
    {
      return LabelItemCount[label];
    }

    public double InverseDocumentCount(string label)
    {
      return LabelItemCount.Where(x => x.Key != label).Sum(x => x.Value);
    }

    protected double LabelCount(string label)
    {
      return !LabelItemCount.ContainsKey(label)
        ? 0
        : LabelItemCount[label];
    }

    protected double TotalCount()
    {
      return LabelItemCount.Sum(c => c.Value);
    }

    public double FeatureProbability(string feature, string label)
    {
      // The total number of times this feature appeared in this
      // label divided by the total number of items in this label
      return FeatureCountForLabel(feature, label)/LabelCount(label);
    }

    public double WeightedProbability(string feature, string label, double weight, double assumedprob)
    {
      // Calculate current probability
      var basicProb = FeatureProbability(feature, label);

      // Count the number of times this feature has appeared in
      // all categories
      var totals = LabelItemCount.Sum(c => FeatureCountForLabel(feature, c.Key));

      // Calculate the weighted average
      var newProb = ((weight * assumedprob) + (totals * basicProb)) / (weight + totals);
      return newProb;
    }

    public double WeightedProbability(string feature, string label)
    {
      return FeatureProbability(feature, label);
      //return WeightedProbability(feature, label, 1.0, 0.5);
    }

    public void SetThreshold(string label, double threshold)
    {
      Thresholds[label] = threshold;
    }

    public double GetThreshold(string label)
    {
      return !Thresholds.ContainsKey(label) ? 0.0 : Thresholds[label];
    }

    public void Log(string text)
    {
      if (_debug)
      {
        Console.WriteLine(text);
      }
    }

    public void SetDebug(bool value)
    {
      _debug = value;
    }
  }
}
