from django.views.generic import TemplateView

class ContactUsView(TemplateView):
    """
    ویوی نمایش صفحه «ارتباط با ما»
    این ویو از TemplateView استفاده می‌کند و فقط قالب را رندر می‌کند.
    """
    template_name = "web/contact_us.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # شماره‌های تماس — می‌توانی این‌ها را از تنظیمات یا مدل‌ها هم بگیری
        context['phone1'] = "+98 21 12345678"
        context['phone2'] = "+98 21 87654321"
        return context
