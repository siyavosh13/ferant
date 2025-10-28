from django.db import models


class Feedback(models.Model):
    RATING_CHOICES = [
        (1, "بد"),
        (2, "معمولی"),
        (3, "خوب"),
        (4, "عالی"),
    ]

    name = models.CharField("نام", max_length=100)
    email = models.EmailField("ایمیل", blank=True)
    message = models.TextField("متن نظر")
    rating = models.PositiveSmallIntegerField("امتیاز", choices=RATING_CHOICES, blank=True, null=True)
    allow_publish = models.BooleanField("اجازه انتشار عمومی", default=False)

    is_public = models.BooleanField("تأیید و نمایش عمومی", default=False)
    created_at = models.DateTimeField("تاریخ ثبت", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "نظر کاربر"
        verbose_name_plural = "نظرات کاربران"

    def __str__(self):
        return f"{self.name} - {self.created_at:%Y-%m-%d}"


class CallRequest(models.Model):
    phone = models.CharField("شماره تماس", max_length=20)
    created_at = models.DateTimeField("تاریخ درخواست", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "درخواست تماس"
        verbose_name_plural = "درخواست‌های تماس"

    def __str__(self):
        return f"{self.phone} — {self.created_at:%Y-%m-%d %H:%M}"


# -------------------------------
# مدل مقاله (Article)
# -------------------------------
class Article(models.Model):
    title = models.CharField("عنوان مقاله", max_length=200)
    content = models.TextField("متن مقاله")
    pdf = models.FileField("فایل PDF", upload_to="articles/pdfs/")

    created_at = models.DateTimeField("تاریخ ثبت", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "مقاله"
        verbose_name_plural = "مقالات"

    def __str__(self):
        return self.title



# ----------------------------مدل کتاب(Article)
# -------------------------------
class Book(models.Model):
    title = models.CharField("عنوان کتاب", max_length=255)
    description = models.TextField("توضیحات", blank=True)
    pdf = models.FileField("فایل PDF", upload_to="books_pdfs/", blank=True, null=True)
    created_at = models.DateTimeField("تاریخ ثبت", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "کتاب"
        verbose_name_plural = "کتاب‌ها"

    def __str__(self):
        return self.title


