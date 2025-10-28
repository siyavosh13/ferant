from django.urls import path
from django.views.generic import RedirectView

from .views.chat import chat_page, chat_api
from .views.speech import speech_to_text
from .views.feedback import FeedbackCreateView, FeedbackThanksView, request_call
from .views.contact import ContactUsView
from .views.articles import article_list, article_detail
from .views.books import book_list, book_detail


  
app_name = "web"

urlpatterns = [
    path("", RedirectView.as_view(pattern_name="web:chat", permanent=False), name="root"),

    # صفحهٔ چت (HTML)
    path("chat/", chat_page, name="chat"),

    # APIها
    path("api/chat/", chat_api, name="chat_api"),
    path("api/speech/", speech_to_text, name="api_speech"),

    # بازخورد
    path("feedback/", FeedbackCreateView.as_view(), name="feedback"),
    path("feedback/thanks/", FeedbackThanksView.as_view(), name="feedback_thanks"),

    # فرم شماره تماس
    path("request-call/", request_call, name="request_call"),

    # صفحه ارتباط با ما
    path('contact/', ContactUsView.as_view(), name='contact_us'),

    # ——— مسیرهای مقاله‌ها ——_
    path("articles/", article_list, name="article_list"),
    path("articles/<int:pk>/", article_detail, name="article_detail"),
    path("books/", book_list, name="book_list"),
    path("books/<int:pk>/", book_detail, name="book_detail"),
]
