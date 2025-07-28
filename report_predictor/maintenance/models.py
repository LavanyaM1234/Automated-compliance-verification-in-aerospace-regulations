from django.db import models

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)



class ReportReview(models.Model):
    report = models.ForeignKey(UploadedFile, on_delete=models.CASCADE)  # Link to the uploaded file
    sentence = models.TextField()  # The sentence being reviewed
    predicted_result = models.CharField(max_length=20)  # 'Compliant' or 'Non-Compliant' prediction
    review_result = models.CharField(max_length=20, choices=[('Compliant', 'Compliant'), ('Non-Compliant', 'Non-Compliant')], null=True, blank=True)
    reviewed = models.BooleanField(default=False)  # Flag if the report has been reviewed by a human
    reviewer_comments = models.TextField(null=True, blank=True)  # Option for reviewer comments
