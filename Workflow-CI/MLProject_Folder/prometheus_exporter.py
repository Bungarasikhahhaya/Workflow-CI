from http.server import BaseHTTPRequestHandler, HTTPServer
import time

# Mencatat waktu pertama kali server dinyalakan
START_TIME = time.time()

class MockMetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4')
            self.end_headers()
            
            # Menghitung uptime dinamis dalam hitungan detik
            current_uptime = time.time() - START_TIME
            
            # 10 METRIKS BERBEDA DENGAN UPDATE UPTIME DINAMIS
            metrics_data = (
                '# HELP heart_disease_predictions_positive_total Total positive predictions\n'
                '# TYPE heart_disease_predictions_positive_total counter\n'
                'heart_disease_predictions_positive_total 42.0\n\n'
                
                '# HELP heart_disease_predictions_negative_total Total negative predictions\n'
                '# TYPE heart_disease_predictions_negative_total counter\n'
                'heart_disease_predictions_negative_total 128.0\n\n'
                
                '# HELP model_accuracy_ratio Current model accuracy simulation\n'
                '# TYPE model_accuracy_ratio gauge\n'
                'model_accuracy_ratio 0.88\n\n'
                
                '# HELP inference_duration_seconds Time taken for model inference\n'
                '# TYPE inference_duration_seconds gauge\n'
                'inference_duration_seconds 0.15\n\n'
                
                '# HELP http_requests_total Total number of HTTP requests\n'
                '# TYPE http_requests_total counter\n'
                'http_requests_total 170.0\n\n'
                
                '# HELP model_confidence_score Average prediction confidence score\n'
                '# TYPE model_confidence_score gauge\n'
                'model_confidence_score 0.92\n\n'
                
                '# HELP system_cpu_usage_percentage Container CPU usage percentage\n'
                '# TYPE system_cpu_usage_percentage gauge\n'
                'system_cpu_usage_percentage 24.5\n\n'
                
                '# HELP system_memory_usage_bytes Container memory usage in bytes\n'
                '# TYPE system_memory_usage_bytes gauge\n'
                'system_memory_usage_bytes 51246000.0\n\n'
                
                '# HELP http_requests_errors_total Total failed HTTP requests\n'
                '# TYPE http_requests_errors_total counter\n'
                'http_requests_errors_total 2.0\n\n'
                
                f'# HELP server_uptime_seconds Server uptime in seconds\n'
                '# TYPE server_uptime_seconds counter\n'
                f'server_uptime_seconds {current_uptime:.1f}\n'
            )
            
            self.wfile.write(metrics_data.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run():
    server_address = ('0.0.0.0', 8082)
    httpd = HTTPServer(server_address, MockMetricsHandler)
    print("==================================================")
    print("Exporter Aktif! Mengeluarkan 10 Metriks Berbeda...")
    print("Silakan cek http://localhost:9090/targets")
    print("==================================================")
    httpd.serve_forever()

if __name__ == '__main__':
    run()