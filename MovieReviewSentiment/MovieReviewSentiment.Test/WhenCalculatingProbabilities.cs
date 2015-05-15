using System;
using System.Collections.Generic;
using MovieReviewSentiment.Classification;
using NUnit.Framework;

namespace MovieReviewSentiment.Test
{
  [TestFixture]
  class WhenCalculatingProbabilities
  {
    private Classifier _cl;

    [SetUp]
    public void Setup()
    {
      Func<string, IList<string>> getFeatures = delegate(string s)
      {
        var docParser = new DocumentParser(true);
        docParser.AddItem(s);
        return docParser.GetFeatures();
      };

      _cl = new Classifier(getFeatures);
    }

    [Test]
    public void It_should_calc_conditional_prob_for_feature_single_word_only()
    {
      _cl.Train("awesome", "good");
      _cl.Train("Awesome", "good");
      _cl.Train("Great", "good");
      _cl.Train("Awesome", "good");

      var p = _cl.FeatureProbability("awesome", "good");

      Assert.AreEqual(0.75, p);
    }

    [Test]
    public void It_should_calc_conditional_prob_for_feature_with_sentences()
    {
      _cl.Train("Nobody owns the water.", "good");
      _cl.Train("the quick rabbit jumps fences", "good");
      _cl.Train("buy pharmaceuticals now", "bad");
      _cl.Train("make quick money at the online casino", "bad");
      _cl.Train("the quick brown fox jumps", "good");

      var p = _cl.FeatureProbability("quick", "good");

      Assert.AreEqual(0.67, Math.Round(p,2));
    }


  }
}
