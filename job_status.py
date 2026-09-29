# job_status.py
"""Tell the user when their OOD job is about to end, or already has.

When PBS kills the job at the end of its wall time (or the node fails, or
the job is qdel'd), the Panel server dies with it. Nothing on the server
can report that, because the server no longer exists. So everything here
runs in the browser:

* JobClock shows the wall time left in the header. It counts down with a
  browser timer, so it keeps going after the server is gone, and it shows
  warning banners 15 and 5 minutes before the end.
* install_disconnect_notice() listens for Bokeh's connection_lost event
  and covers the page with a notice. The page would otherwise look fine
  while it silently stops responding.

The job's end time comes from INFERSTUDIO_JOB_END (Unix seconds), which
the OOD app's template/script.sh.erb (NCAR/bc_InferStudio) sets from the
requested wall time. When it is unset, as in a local `panel serve`,
there's no clock. The disconnect notice still appears.
"""
import os

import panel as pn
import param
from bokeh.models import CustomJS
from panel.reactive import ReactiveHTML

# Minutes before the end of the job at which a warning banner appears.
WARN_MINUTES = (15, 5)


def job_end_time():
    """Return the job's end as Unix seconds, or None when it is unknown."""
    try:
        return float(os.environ["INFERSTUDIO_JOB_END"])
    except (KeyError, ValueError):
        return None


# Defines window.inferstudioBanner(html, kind) once per page. The banner is
# attached to document.body, not inside any Panel component, so it overlays
# the whole app. It is shared by JobClock and the disconnect notice, so the
# final "job ended" message replaces any earlier warning.
_BANNER_JS = """
if (!window.inferstudioBanner) {
  window.inferstudioBanner = function (html, kind) {
    var el = document.getElementById('inferstudio-job-banner');
    if (!el) {
      el = document.createElement('div');
      el.id = 'inferstudio-job-banner';
      document.body.appendChild(el);
    }
    var fatal = kind === 'ended';
    el.style.cssText = fatal
      ? 'position:fixed;inset:0;z-index:100000;display:flex;align-items:center;' +
        'justify-content:center;background:rgba(9,20,34,0.85);'
      : 'position:fixed;top:12px;left:50%;transform:translateX(-50%);' +
        'z-index:100000;';
    var box = 'max-width:560px;padding:18px 22px;border-radius:8px;' +
      'font:15px/1.45 sans-serif;box-shadow:0 4px 18px rgba(0,0,0,0.35);' +
      (fatal ? 'background:#fff;color:#1a1a1a;'
             : 'background:#fff4d6;color:#4a3500;border:1px solid #e0b43a;');
    var close = fatal ? '' :
      '<button style="float:right;margin-left:14px;border:none;' +
      'background:none;font-size:18px;cursor:pointer;color:inherit" ' +
      'onclick="this.closest(\\'#inferstudio-job-banner\\').remove()" ' +
      'aria-label="Dismiss">&times;</button>';
    el.innerHTML = '<div role="alert" style="' + box + '">' + close + html + '</div>';
  };
}
"""

# Lets the disconnect notice say *why* the connection dropped: once the end
# time has passed, the wall time is the cause.
_ENDED_JS = """
var end = window.inferstudioJobEnd;
var expired = end != null && Date.now() / 1000 >= end - 30;
var html = expired
  ? '<b style="font-size:17px">Your InferStudio job has ended</b><br><br>' +
    'It reached the end of the wall time you requested, so the server has ' +
    'shut down and this page no longer responds.<br><br>' +
    'Output files already written to disk are kept. To keep working, ' +
    'start a new InferStudio session from NCAR Open OnDemand, and request ' +
    'more wall time if you need it.'
  : '<b style="font-size:17px">Lost connection to InferStudio</b><br><br>' +
    'The server stopped responding. The job may have been stopped or ' +
    'cancelled, or your network connection dropped.<br><br>' +
    'Check the job\\'s status in NCAR Open OnDemand under ' +
    '<i>My Interactive Sessions</i>. If it is still running, reload this ' +
    'page. If not, start a new session.';
window.inferstudioBanner(html, 'ended');
"""


class JobClock(ReactiveHTML):
    """Header readout of the job's remaining wall time."""

    end_time = param.Number(default=None, allow_None=True, doc="""
        Unix seconds at which the job ends.""")

    warn_minutes = param.List(default=list(WARN_MINUTES))

    _template = (
        '<span id="clock" title="Wall time left before this job ends" '
        'style="color:#DFEFF6;font-size:14px;font-weight:500;'
        'white-space:nowrap;"></span>'
    )

    _scripts = {
        "render": _BANNER_JS + """
          window.inferstudioJobEnd = data.end_time;
          state.warned = {};
          function fmt(s) {
            var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
            return h > 0 ? h + 'h ' + m + 'm' : m + 'm';
          }
          function tick() {
            var left = data.end_time - Date.now() / 1000;
            if (left <= 0) {
              clock.textContent = 'Job ended';
              clock.style.color = '#ff8a80';
              clearInterval(state.timer);
              return;
            }
            clock.textContent = 'Job time left: ' + fmt(left);
            var mins = left / 60;
            clock.style.color = mins <= Math.min(...data.warn_minutes)
              ? '#ff8a80' : mins <= Math.max(...data.warn_minutes)
              ? '#ffd166' : '#DFEFF6';
            for (var w of data.warn_minutes) {
              if (mins <= w && !state.warned[w]) {
                state.warned[w] = true;
                window.inferstudioBanner(
                  '<b>About ' + Math.max(1, Math.ceil(mins)) + ' minutes of ' +
                  'job time left.</b> InferStudio will shut down when your ' +
                  'job reaches its wall time. Unfinished inference runs ' +
                  'will stop, and any work not yet saved to disk will be lost.',
                  'warn');
              }
            }
          }
          tick();
          state.timer = setInterval(tick, 10000);
        """,
        "remove": "clearInterval(state.timer)",
    }


def job_clock(**params):
    """Return a JobClock for this job, or None outside an OOD job."""
    end = job_end_time()
    if end is None:
        return None
    return JobClock(end_time=end, **params)


def install_disconnect_notice(doc=None):
    """Cover the page with a notice when the browser loses the server.

    Must be called while the session's document is being built (for
    example from build_app), so the callback is sent with the document.
    """
    doc = doc or pn.state.curdoc
    if doc is None:
        return
    doc.js_on_event("connection_lost", CustomJS(code=_BANNER_JS + _ENDED_JS))
