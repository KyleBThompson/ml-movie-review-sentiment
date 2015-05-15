using System;
using System.Collections.Generic;
using MovieReviewSentiment.Classification;
using NUnit.Framework;

namespace MovieReviewSentiment.Test
{
  [TestFixture]
  public class WhenTrainingTheClassifier
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
    public void It_should_be_trainable_with_one_review()
    {
      _cl.Train("Worth every penny", "good");
      Assert.AreEqual(1.0, _cl.FeatureCountForLabel("penny", "good"));
      Assert.AreEqual(1.0, _cl.FeatureCountForLabel("every", "good"));
      Assert.AreEqual(1.0, _cl.FeatureCountForLabel("penny", "good"));
    }

    [Test]
    public void It_should_be_trainable_with_multiple_review_same_category()
    {
      _cl.Train("Great", "good");
      _cl.Train("Great", "good");
      Assert.AreEqual(2, _cl.FeatureCountForLabel("Great", "good"));
    }

    [Test]
    public void It_should_be_trainable_with_same_feature_diff_category()
    {
      _cl.Train("Great", "good");
      _cl.Train("Great", "bad");
      Assert.AreEqual(1, _cl.FeatureCountForLabel("Great", "good"));
      Assert.AreEqual(1, _cl.FeatureCountForLabel("Great", "bad"));
    }

    [Test]
    public void It_should_be_trainable_with_multiple_review_diff_category()
    {
      _cl.Train("Great movie", "good");
      _cl.Train("Great movie", "good");
      _cl.Train("Terrible movie", "bad");
      _cl.Train("Terrible movie", "bad");
      Assert.AreEqual(2, _cl.FeatureCountForLabel("Great", "good"));
      Assert.AreEqual(2, _cl.FeatureCountForLabel("Terrible", "bad"));
      Assert.AreEqual(2, _cl.FeatureCountForLabel("movie", "good"));
      Assert.AreEqual(2, _cl.FeatureCountForLabel("movie", "bad"));
    }

    [Test]
    public void It_should_not_be_case_sensitive()
    {
      _cl.Train("Great", "good");
      _cl.Train("great", "good");
      Assert.AreEqual(2, _cl.FeatureCountForLabel("Great", "good"));
    }

  }
}
