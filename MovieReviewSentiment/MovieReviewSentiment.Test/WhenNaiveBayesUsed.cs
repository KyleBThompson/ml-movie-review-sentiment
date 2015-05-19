using System;
using System.Collections.Generic;
using MovieReviewSentiment.Classification;
using NUnit.Framework;

namespace MovieReviewSentiment.Test
{
  [TestFixture]
  public class WhenNaiveBayesUsed
  {
    private NaiveBayes _cl;

    [SetUp]
    public void Setup()
    {
      Func<string, IList<string>> getFeatures = delegate(string s)
      {
        var docParser = new DocumentParser();
        docParser.AddItem(s);
        return docParser.GetFeatures();
      };

      _cl = new NaiveBayes(getFeatures);
    }

    [Test]
    public void It_should_classify()
    {
      _cl.Train("Nobody owns the water.", "good");
      _cl.Train("the quick rabbit jumps fences", "good");
      _cl.Train("buy pharmaceuticals now", "bad");
      _cl.Train("make quick money at the online casino", "bad");
      _cl.Train("the quick brown fox jumps", "good");

      var classification = _cl.Classify("quick rabbit");
      Console.WriteLine(classification.Probability);
      Assert.AreEqual("good", classification.Label);

      classification = _cl.Classify("money");
      Console.WriteLine(classification.Probability);
      Assert.AreEqual("bad", classification.Label);

    }

    [Test]
    public void Log_subtraction_should_equal_float_division()
    {
      const double probability = .80;
      const double inverseProbability = 1 - probability;
      const double quotient1 = inverseProbability / probability;
      var quotient2 = Math.Exp(Math.Log(inverseProbability) - Math.Log(probability));
      Assert.AreEqual(quotient1, quotient2);
    }
  }
}
