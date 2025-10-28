# web/views/feedback.py
from django.urls import reverse_lazy
from django.views.generic.edit import CreateView
from django.views.generic.list import ListView
from django.contrib import messages
from django.shortcuts import render

# چون این فایل داخل web/views/ قرار دارد، برای دسترسی به models و forms در سطح web از .. استفاده می‌کنیم
from ..models import Feedback, CallRequest
from ..forms import FeedbackForm, CallRequestForm


class FeedbackCreateView(CreateView):
    model = Feedback
    form_class = FeedbackForm
    template_name = "feedback/feedback_form.html"
    success_url = reverse_lazy("feedback_thanks")  # اگر در پروژه‌ات namespace اضافه نکردی، همین باقی بمونه

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


def request_call(request):
    """
    فرم ثبت شماره تماس کاربر. بعد از ثبت روی همان صفحه پیام موفقیت نمایش داده می‌شود.
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

    return render(request, "web/request_call.html", {"form": form, "success_message": success_message})
