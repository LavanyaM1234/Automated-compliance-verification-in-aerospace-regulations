# from django.urls import path
# from . import views

# urlpatterns = [
#     path('',views.upload_file,name='upload_file'),
#     path('upload/', views.upload_file, name='upload_file'),
#     path('download/<int:file_id>/', views.download_file, name='download_file'),
# ]


from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home_page, name='home_page'),
    path('upload/', views.upload_file, name='upload_file'),
    path('download/<int:file_id>/', views.download_file, name='download_file'),
    path('review/<int:review_id>/', views.review_report, name='review_report'),
    path('review_done/', views.review_done, name='review_done'),
    path('review_reports/', views.review_reports, name='review_reports'),
    path('home_page/', views.home_page, name='home_page'),
    path('uploadp/', views.upload, name='upload123'),
    path('header_page/', views.header_page, name='header_page'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
