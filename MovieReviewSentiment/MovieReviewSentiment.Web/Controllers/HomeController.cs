using System.Net;
using System.Net.Http;
using System.Web.Mvc;

namespace MovieReviewSentiment.Web.Controllers
{
  public class HomeController : Controller
  {
    public ActionResult Index(string returnUrl)
    {
      ViewBag.ReturnUrl = returnUrl;
      return View();
    }

  }
}