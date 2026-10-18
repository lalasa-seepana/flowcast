"""Render all required Streamlit views against persisted real results."""
import unittest
import hashlib
import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest
from src.utils import ROOT


class DashboardTests(unittest.TestCase):
    def test_nine_views_and_role_defaults(self):
        app = AppTest.from_file(str(ROOT / 'dashboard/app.py'), default_timeout=120)
        app.run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertEqual(app.sidebar.radio[0].value, 'Congestion heatmap')
        views = list(app.sidebar.radio[0].options)
        self.assertEqual(len(views), 9)
        for view in views:
            with self.subTest(view=view):
                app.sidebar.radio[0].set_value(view).run()
                self.assertFalse(app.exception, str(app.exception))
                self.assertFalse(app.error, str(app.error))
        app.sidebar.selectbox[0].set_value('Transport Planner').run()
        self.assertEqual(app.sidebar.radio[0].value, 'Road comparison')
        self.assertFalse(app.exception, str(app.exception))

    def test_prediction_action_uses_saved_models(self):
        app = AppTest.from_file(str(ROOT / 'dashboard/app.py'), default_timeout=180).run()
        button = next(b for b in app.button if b.label == 'Run prediction')
        button.click().run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertFalse(app.error, str(app.error))
        self.assertEqual(len(app.session_state['requested_forecast']), 100)

    def test_upload_prediction_and_report_export(self):
        source = pd.read_csv(ROOT / 'data/raw/traffic_sensor_log.csv')
        incoming = source.loc[source.date.eq('2025-05-30')].tail(100)
        upload = io.BytesIO(incoming.to_csv(index=False).encode())
        # AppTest has no file-upload action; supply real CSV bytes at that boundary.
        with patch('streamlit.file_uploader', return_value=upload), \
                patch('streamlit.download_button', wraps=st.download_button) as download:
            app = AppTest.from_file(str(ROOT / 'dashboard/app.py'), default_timeout=180).run()
            next(b for b in app.button if b.label == 'Validate upload and predict').click().run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertFalse(app.error, str(app.error))
        result = app.session_state['requested_forecast']
        self.assertEqual(len(result), 100)
        self.assertEqual(set(result.upload_sha256),
                         {hashlib.sha256(upload.getvalue()).hexdigest()})
        exports = {call.args[0]: call.args[1] for call in download.call_args_list}
        self.assertIn('Download upload predictions', exports)
        self.assertIn('Download complete upload validation audit', exports)
        report = exports['Export selected-period report']
        self.assertIn('2025-05-25 to 2025-05-31', report)
        self.assertIn('NL-001', report)
        self.assertIn('observed_windows', report)

    def test_training_launch_and_status_controls(self):
        # Check the UI launch contract without replacing the delivered models.
        # Actual training stages are checked separately on real source records.
        process = Mock()
        process.poll.return_value = None
        original_open = Path.open
        with tempfile.TemporaryDirectory(prefix='flowcast-training-ui-') as temporary:
            def open_log(path, *args, **kwargs):
                if path == ROOT / 'reports/retraining.log':
                    path = Path(temporary) / 'retraining.log'
                return original_open(path, *args, **kwargs)
            with patch.object(Path, 'open', open_log), \
                    patch('subprocess.Popen', return_value=process) as launch:
                app = AppTest.from_file(str(ROOT / 'dashboard/app.py'), default_timeout=120).run()
                next(b for b in app.button if b.label == 'Retrain from original inputs').click().run()
                self.assertFalse(app.exception, str(app.exception))
                self.assertEqual(launch.call_args.args[0], [sys.executable, '-m', 'src.utils'])
                self.assertEqual(launch.call_args.kwargs['cwd'], ROOT)
                next(b for b in app.button if b.label == 'Check training status').click().run()
                self.assertTrue(any('Training is running' in item.value for item in app.info))
                process.poll.return_value = 0
                process.returncode = 0
                next(b for b in app.button if b.label == 'Check training status').click().run()
                self.assertTrue(any('Training completed' in item.value for item in app.success))
                self.assertFalse(app.exception, str(app.exception))


if __name__ == '__main__':
    unittest.main()
