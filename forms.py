from django import forms
from django.core.validators import RegexValidator
from .models import Feedback, CallRequest  # اضافه کردن مدل Book

# -------------------------------
# فرم ارسال بازخورد (Feedback)
# -------------------------------
class FeedbackForm(forms.ModelForm):
    honeypot = forms.CharField(required=False, widget=forms.HiddenInput)

    class Meta:
        model = Feedback
        fields = ["name", "email", "message", "rating", "allow_publish"]
        widgets = {
            "name": forms.TextInput(attrs={"placeholder": "نام شما"}),
            "email": forms.EmailInput(attrs={"placeholder": "اختیاری"}),
            "message": forms.Textarea(attrs={"rows": 4, "placeholder": "نظر شما..."}),
        }

    def clean_honeypot(self):
        """جلوگیری از ارسال اسپم"""
        if self.cleaned_data.get("honeypot"):
            raise forms.ValidationError("مشکل در ارسال فرم.")
        return ""

# -------------------------------
# فرم درخواست تماس (Call Request)
# -------------------------------
class CallRequestForm(forms.ModelForm):
    phone = forms.CharField(
        max_length=20,
        label="شماره تماس شما",
        validators=[
            RegexValidator(
                regex=r'^[0-9+\-\s]{9,20}$',
                message="لطفاً شماره تماس را فقط با ارقام، فاصله یا + وارد کنید (حداقل ۹ رقم)."
            )
        ],
        widget=forms.TextInput(attrs={
            "placeholder": "مثال: +98 912 345 6789",
            "class": "form-control",
        })
    )

    class Meta:
        model = CallRequest
        fields = ["phone"]

