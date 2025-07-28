import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import render
from .models import UploadedFile
from .forms import UploadFileForm
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import os
import csv
from django.conf import settings

MODEL_PATH = "D:/aiml_partb_3/report_predictor/maintenance/saved_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# ================== 1. Load the Regulations CSV File ================== #
regulations_path = 'D:/aiml_partb_3/report_predictor/maintenance/regulations.csv'
regulations_df = pd.read_csv(regulations_path)

def classify_text(text):
    """Tokenize input text and classify using the trained model."""
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    return "Compliant" if prediction == 0 else "Non-Compliant"


def review_done(request):
    return render(request, 'review_done.html')  # Render a template confirming the review


def extract_summary(file_content):
    """Extract the summary section from the report content, including numbered points."""
    # Make the search for 'Summary:' case-insensitive
    summary_start = file_content.lower().find('summary:')
    
    if summary_start == -1:
        print("No 'Summary:' section found in the file content.")  # Debugging line
        return ""  # Return empty string if 'Summary:' is not found
    
    # Extract everything after 'Summary:'
    summary_text = file_content[summary_start + 8:].strip()  # Skip past "Summary:"
    
    # Use regex to match numbered points (e.g., 1), 2), 3)) and capture until no more numbered points are found.
    numbered_points = re.findall(r'^\s*\d+\)\s+.*', summary_text, re.MULTILINE)

    if not numbered_points:
        print("No numbered points found in the summary.")  # Debugging line

    # Join the matched numbered points into a single string.
    summary_text = "\n".join(numbered_points).strip()
    
    return summary_text


def evaluate_summary(summary, uploaded_file):
    """Evaluate each numbered point in the summary for compliance and similarity check."""
    print("Summary to evaluate:", summary)  # Debugging line
    results = []

    # Evaluate each numbered point
    for point in summary.splitlines():  # Split the summary into individual lines (numbered points)
        point = point.strip()  # Remove any leading/trailing whitespace
        if point:  # Ensure the point is not empty
            prediction = classify_text(point)
            print(f"Classified: {point} -> {prediction}")  # Debugging line
            result = {
                "sentence": point,
                "result": prediction
            }

            # Save prediction to ReportReview for human review
            review = ReportReview.objects.create(
                report=uploaded_file,
                sentence=point,
                predicted_result=prediction,
                reviewed=False  # Mark as not reviewed yet
            )

            # If prediction is Non-Compliant, perform similarity check with regulations
            if prediction == "Non-Compliant":
                # Combine the non-compliant report with regulation details for vectorization
                texts = [point] + regulations_df["Regulation details"].tolist()

                # Create a TF-IDF Vectorizer to calculate similarity
                vectorizer = TfidfVectorizer(stop_words="english")
                tfidf_matrix = vectorizer.fit_transform(texts)

                # The first row is the non-compliant report, and the rest are regulations
                non_compliant_tfidf = tfidf_matrix[0:1, :]
                regulations_tfidf = tfidf_matrix[1:, :]

                # Compute cosine similarity between the non-compliant report and all regulations
                similarity_matrix = cosine_similarity(non_compliant_tfidf, regulations_tfidf)

                # Find the regulation with the highest similarity score
                similarities = similarity_matrix[0]
                max_similarity_index = similarities.argmax()
                best_regulation = regulations_df.iloc[max_similarity_index]

                # Add regulation details and similarity score to the result
                result.update({
                    "regulation_title": best_regulation['Regulation Title'],
                    "regulation_number": best_regulation['Regulation number'],
                    "regulation_details": best_regulation['Regulation details'],
                })

                print(f"Non-Compliant Report: {point}")
                print(f"Most Similar Regulation: {best_regulation['Regulation Title']}")
                print(f"Regulation Number: {best_regulation['Regulation number']}")
                print(f"Regulation Details: {best_regulation['Regulation details']}")
                print("-" * 100)

            results.append(result)

    return results

from django.shortcuts import render, get_object_or_404, redirect
from .models import ReportReview

from django.contrib import messages

def review_report(request, review_id):
    """Review a report prediction and allow human reviewer to correct the result."""
    review = get_object_or_404(ReportReview, id=review_id)

    if request.method == 'POST':
        updated_result = request.POST.get('review_result')
        reviewer_comments = request.POST.get('reviewer_comments')

        # Check if this sentence is already reviewed
        existing_review = ReportReview.objects.filter(sentence=review.sentence).first()

        # If the sentence is already reviewed, update the existing review instead of creating a new one
        if existing_review:
            existing_review.review_result = updated_result
            existing_review.reviewer_comments = reviewer_comments
            existing_review.reviewed = True
            existing_review.save()
            review = existing_review  # Assign the updated review to the local variable
        else:
            # Update the review with the new result and comments
            review.review_result = updated_result
            review.reviewer_comments = reviewer_comments
            review.reviewed = True
            review.save()

        # Save the changes to a CSV file
        save_review_to_csv(review)

        # Success message after reviewing
        messages.success(request, 'Review submitted successfully.')

        return redirect('review_reports')  # Redirect back to the list of reports for review

    return render(request, 'review_report.html', {'review': review})

def save_review_to_csv(review):
    """Save reviewed data into a CSV file."""
    # Define the file path for saving reviewed reports (ensure it's writable)
    csv_file_path = os.path.join(settings.BASE_DIR, 'reviewed_reports.csv')

    # Check if the file exists; if not, create it and add headers
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        fieldnames = ['report_id', 'sentence', 'predicted_result', 'review_result', 'reviewer_comments', 'reviewed']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header only if the file doesn't exist
        if not file_exists:
            writer.writeheader()

        # Write the review data
        writer.writerow({
            'report_id': review.report.id,
            'sentence': review.sentence,
            'predicted_result': review.predicted_result,
            'review_result': review.review_result,
            'reviewer_comments': review.reviewer_comments,
            'reviewed': review.reviewed
        })

def review_reports(request):
    """Display reports that need human review."""
    pending_reviews = ReportReview.objects.filter(reviewed=False)  # Fetch unreviewed reports

    return render(request, 'review_reports.html', {'pending_reviews': pending_reviews})

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save()

            # Read file contents
            file_path = uploaded_file.file.path
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            # Extract summary from file content
            summary = extract_summary(file_content)

            # Pass the uploaded_file object to evaluate_summary
            results = evaluate_summary(summary, uploaded_file)  # Pass uploaded_file here

            # Return result to the template
            return render(request, 'upload_file.html', {
                'form': form,
                'files': UploadedFile.objects.all(),
                'results': results,
                'file_name': uploaded_file.file.name
            })
    else:
        form = UploadFileForm()

    return render(request, 'upload_file.html', {'form': form, 'files': UploadedFile.objects.all()})




def download_file(request, file_id):
    uploaded_file = UploadedFile.objects.get(pk=file_id)
    response = HttpResponse(uploaded_file.file, content_type='application/force-download')
    response['Content-Disposition'] = f'attachment; filename="{uploaded_file.file.name}"'
    return response

def home_page(request):
    return render(request, 'main.html')

def upload(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']

        # Save the uploaded file as an UploadedFile instance
        uploaded_file_instance = UploadedFile(file=uploaded_file)
        uploaded_file_instance.save()  # Save the file to the database

        # Read the file contents (assuming it's a text-based file like .txt, .pdf, or .docx)
        file_path = f'media/{uploaded_file.name}'
        
        # Save the file to the media directory (optional)
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        # Open and read the file content (you can adapt this for different file types like PDF or DOCX)
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        
        # Extract summary or perform compliance analysis
        summary = extract_summary(file_content)  # Function to extract summary from file content
        
        # Pass the uploaded file instance to evaluate and get classification results
        results = evaluate_summary(summary, uploaded_file_instance)  # Use the uploaded file instance here

        # Create a ReportReview object and associate the uploaded file
        report_review = ReportReview(report=uploaded_file_instance)  # Assign the file to the report field
        report_review.save()  # Save the review instance

        # Return results to the template for display
        return render(request, 'upload.html', {
            'form': UploadFileForm(),
            'results': results,
            'file_name': uploaded_file.name
        })
    
    return render(request, 'upload.html')

def header_page(request):
    return render(request, 'header.html')

