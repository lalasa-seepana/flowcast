"""Render all required Streamlit views against persisted real results."""
import unittest
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


if __name__ == '__main__':
    unittest.main()
