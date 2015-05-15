using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace MovieReviewSentiment.Classification
{
  public class DocumentParser
  {
    private readonly IList<string> _features;
    private readonly Stemmer _stemmer;
    private readonly bool _suppressStemming;

    public DocumentParser()
    {
      _features = new List<string>();
      _stemmer = new Stemmer();
    }

    public DocumentParser(bool suppressStemming): this()
    {
      _suppressStemming = suppressStemming;
    }

    public IList<string> GetFeatures()
    {
      return _features;
    }

    private IEnumerable<string> Tokenize(string text)
    {
      var tokens = text.Split(' ')
        .Select(x => x.ToLower())
        .Select(x => x.Replace("\"", ""))
        .Select(x => x.Replace("'", ""))
        .Select(x => Regex.Replace(x, @"\W", " "))
        .Select(x => Regex.Replace(x, @"\s+", " "))
        .Select(x => x.Trim())
        .Where(x => x.Length > 0)
        .Select(StemWord)
        .Distinct().ToList();

      return ApplyNegations(tokens);
    }

    private static IEnumerable<string> ApplyNegations(List<string> tokens)
    {
      var negations =
        new Regex(
          @"^(never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|
              shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$");
      
      for (var i = 0; i < tokens.Count; i++)
      {
        if (!negations.IsMatch(tokens[i])) continue;
        if (i < tokens.Count - 1)
        {
          tokens[i + 1] = "!" + tokens[i + 1];
        }
        if (i > 0)
        {
          tokens[i - 1] = "!" + tokens[i - 1];
        }
      }

      return tokens;
    }

    private string StemWord(string word)
    {
      if (_suppressStemming)
      {
        return word;
      }
      foreach (var c in word)
      {
        _stemmer.add(c);
      }
      _stemmer.stem();
      return _stemmer.ToString();
    }

    public void AddItem(string text)
    {
      foreach (var f in Tokenize(text))
      {
        _features.Add(f);
      }
    }
  }
}
