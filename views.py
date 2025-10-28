from django.urls import reverse_lazy
from django.views.generic.edit import CreateView
from django.views.generic.list import ListView
from django.views.generic import TemplateView
from django.contrib import messages
from django.shortcuts import render, get_object_or_404
from .models import Feedback, CallRequest, Article
from .forms import FeedbackForm, CallRequestForm


# ——— Feedback Views ———
class FeedbackCreateView(CreateView):
    model = Feedback
    form_class = FeedbackForm
    template_name = "feedback/feedback_form.html"
    success_url = reverse_lazy("web:feedback_thanks")

    def form_valid(self, form):
        messages.success(self.request, "نظر شما با موفقیت ارسال شد. سپاس از بازخورد شما!")
        return super().form_valid(form)


class FeedbackThanksView(ListView):
    model = Feedback
    template_name = "feedback/feedback_thanks.html"
    context_object_name = "items"
    paginate_by = 5

    def get_queryset(self):
        return Feedback.objects.filter(is_public=True)


# ——— Call Request View ———
def request_call(request):
    """
    فرم ثبت شماره تماس کاربر.
    بعد از ثبت، روی همان صفحه پیام موفقیت نمایش داده می‌شود.
    """
    success_message = None

    if request.method == "POST":
        form = CallRequestForm(request.POST)
        if form.is_valid():
            form.save()
            success_message = "شماره تماس شما با موفقیت ثبت شد؛ به زودی با شما تماس می‌گیریم."
            form = CallRequestForm()  # فرم خالی بعد از ثبت
    else:
        form = CallRequestForm()

    return render(request, "web/request_call.html", {
        "form": form,
        "success_message": success_message
    })


# ——— Contact Us View ———
class ContactUsView(TemplateView):
    """
    ویوی نمایش صفحه «ارتباط با ما»
    این ویو فقط قالب web/contact_us.html را رندر می‌کند.
    """
    template_name = "web/contact_us.html"


# ——— Article Views ———
def article_list(request):
    """
    نمایش لیست تمام مقاله‌ها
    """
    articles = Article.objects.all()
    return render(request, "web/article_list.html", {"articles": articles})


def article_detail(request, pk):
    """
    نمایش جزئیات یک مقاله همراه با PDF
    """
    article = get_object_or_404(Article, pk=pk)
    return render(request, "web/article_detail.html", {"article": article})
