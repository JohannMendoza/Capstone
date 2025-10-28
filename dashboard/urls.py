from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from .views import (
    register_view, 
    verify_email_view, 
    login_view,
    logout_view,
    home_view, 
    admin_dashboard,
    plant_inventory, 
    add_plant, 
    update_plant, 
    delete_plant,
    track_plant_health,
    history_view,
    analysis_detail_view,
    client_dashboard, 
    user_list, 
    update_user_view,
    reports_view,
    export_csv,
    export_pdf,
    CustomPasswordResetView,
    CustomPasswordResetDoneView,
    CustomPasswordResetConfirmView,
    CustomPasswordResetCompleteView,
)

urlpatterns = [
    path('', home_view, name='home'),
    path('register/', register_view, name='register'),
    path("verify/<uidb64>/<token>/", verify_email_view, name="verify_email"),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('admin_dashboard/', admin_dashboard, name='admin_dashboard'),
    
    path('inventory/', plant_inventory, name="inventory"),  
    path('inventory/add/', add_plant, name="add_plant"),  
    path('inventory/update/<int:plant_id>/', update_plant, name="update_plant"),  
    path('inventory/delete/<int:plant_id>/', delete_plant, name="delete_plant"),  
    
    path('track-plant-health/', track_plant_health, name='track_plant_health'),
    path("reports/", reports_view, name="reports"),
    path("reports/export/csv/", export_csv, name="export_csv"),
    path("reports/export/pdf/", export_pdf, name="export_pdf"),
    path('client_dashboard/', client_dashboard, name='client_dashboard'),
    path('users/', user_list, name='user_list'),
    path("update-user/<int:user_id>/", update_user_view, name="update_user"),
    path("delete-user/<int:user_id>/", user_list, name="delete_user"),
    
    path("password_reset/", CustomPasswordResetView.as_view(), name="password_reset"),
    path("password_reset/done/", CustomPasswordResetDoneView.as_view(), name="password_reset_done"),
    path("reset/<uidb64>/<token>/", CustomPasswordResetConfirmView.as_view(), name="password_reset_confirm"),
    path("reset/done/", CustomPasswordResetCompleteView.as_view(), name="password_reset_complete"),
    
    path("detector/", views.detector, name="detector"),
    path('predict/', views.predict, name='predict'),
    path('tree-analysis/', views.new_tree_analysis, name='new_tree_analysis'),
    path('tree-analysis/<int:analysis_id>/', views.tree_analysis, name='tree_analysis'),
    path('tree-analysis/<int:analysis_id>/complete/', views.complete_analysis, name='complete_analysis'),
    path('tree-analysis/<int:analysis_id>/clear/', views.clear_leaves, name='clear_leaves'),
    path('leaf/<int:leaf_id>/remove/', views.remove_leaf, name='remove_leaf'),
    path('history/', history_view, name='history'),
    path('analysis/<int:analysis_id>/', views.analysis_detail, name='analysis_detail'),
    path('analysis/<int:analysis_id>/delete/', views.delete_analysis, name='delete_analysis'),
    path('analysis/<int:analysis_id>/export/pdf/', views.export_analysis_pdf, name='export_analysis_pdf'),
    path('analyses/delete/', views.delete_multiple_analyses, name='delete_multiple_analyses'),
    path('analyses/export/', views.export_all_analyses, name='export_all_analyses'),
    path('analysis/<int:analysis_id>/detail/', analysis_detail_view, name='analysis_detail'),
    
    path('save-analysis/', views.save_analysis, name='save_analysis'),
    
    path('pest-detector/', views.pest_detector, name='pest_detector'),
    path('pest-predict/', views.pest_predict, name='pest_predict'),
    path('save-pest-results/', views.save_pest_results, name='save_pest_results'),
    
    path('pest-session/<int:session_id>/delete/', views.delete_pest_session, name='delete_pest_session'),
    path('pest-session/<int:session_id>/export/csv/', views.export_pest_session_csv, name='export_pest_session_csv'),
    
    path('analyses/export/multiple/', views.export_multiple_analyses, name='export_multiple_analyses'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
