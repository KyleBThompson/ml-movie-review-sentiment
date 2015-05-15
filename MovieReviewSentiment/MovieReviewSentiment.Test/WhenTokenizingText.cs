using System.Linq;
using MovieReviewSentiment.Classification;
using NUnit.Framework;

namespace MovieReviewSentiment.Test
{
  [TestFixture]
  public class WhenTokenizingText
  {
    private DocumentParser _docParser;

    [SetUp]
    public void Setup()
    {
      _docParser = new DocumentParser();
    }

    [Test]
    public void It_should_split_into_words()
    {
      const string text = "the best damn movie ever";
      _docParser.AddItem(text);
      var words = _docParser.GetFeatures();

      Assert.AreEqual(5, words.Count);
    }

    [Test]
    public void It_should_convert_to_lower_case()
    {
      const string text = "Blah Ha";
      _docParser.AddItem(text);
      var words = _docParser.GetFeatures();

      Assert.AreEqual(2, words.Count);
      var resultingText = string.Join(" ", words.ToArray());
      Assert.AreEqual("blah ha", resultingText);
    }

    [Test]
    public void It_should_split_into_unique_words()
    {
      const string text = "saw most terrible terrible terrible movie";
      _docParser.AddItem(text);
      var words = _docParser.GetFeatures();

      Assert.AreEqual(4, words.Count);
    }

    [Test]
    public void It_should_handle_non_word_characters_as_word()
    {
      const string text = "just ... meh";
      _docParser.AddItem(text);
      var words = _docParser.GetFeatures();
      Assert.AreEqual(2, words.Count);
    }

    [Test]
    public void It_should_handle_non_word_characters_in_words()
    {
      const string text = "uh-oh";
      _docParser.AddItem(text);
      var words = _docParser.GetFeatures();
      Assert.AreEqual(1, words.Count);
      Assert.AreEqual("uh oh", words[0]);
    }



  }
}
