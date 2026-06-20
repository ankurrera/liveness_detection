// AuraSense - The frontend brains for our workforce monitor
document.addEventListener("DOMContentLoaded", () => {
    let employeeId = 1;
    let timeSplitChart = null;

    // grab all our dom elements so we don't have to keep hunting for them later
    const employeeSelectEl = document.getElementById("employeeSelect");
    const metricStatusEl = document.getElementById("metric-status");
    const blockStatusEl = document.getElementById("block-status");
    
    const metricWorkingEl = document.getElementById("metric-working");
    const metricIdleEl = document.getElementById("metric-idle");
    const metricAbsentEl = document.getElementById("metric-absent");
    const metricProductivityEl = document.getElementById("metric-productivity");

    const feedModeEl = document.getElementById("feed-mode");
    const feedConfidenceEl = document.getElementById("feed-confidence");
    
    const summaryFirstEl = document.getElementById("summary-first");
    const summaryLastEl = document.getElementById("summary-last");
    const summaryMonitoredEl = document.getElementById("summary-monitored");
    const summaryProdPctEl = document.getElementById("summary-prod-pct");

    const timelineListEl = document.getElementById("timeline-list");
    const logsTableBodyEl = document.getElementById("logs-table-body");

    // same thing for the debug panel stuff
    const debugRawScoreEl = document.getElementById("debug-raw-score");
    const debugSmoothedScoreEl = document.getElementById("debug-smoothed-score");
    const debugThresholdEl = document.getElementById("debug-threshold");
    const debugIdleCdEl = document.getElementById("debug-idle-cd");
    const debugEpsilonEl = document.getElementById("debug-epsilon");

    // kick things off
    initChart();
    
    // set up the dropdown and logic for switching who we're watching
    async function initEmployeeSelector() {
        try {
            // 1. ask the backend who we are currently watching
            const activeRes = await fetch("/api/active_employee");
            if (activeRes.ok) {
                const activeData = await activeRes.json();
                employeeId = activeData.active_employee_id || 1;
            }
            
            // 2. get the list of everyone else so we can populate the dropdown
            const response = await fetch("/api/employees");
            if (response.ok) {
                const employees = await response.json();
                employeeSelectEl.innerHTML = employees.map(emp => {
                    const isSelected = emp.employee_id === employeeId ? "selected" : "";
                    return `<option value="${emp.employee_id}" ${isSelected}>${emp.name} (${emp.department})</option>`;
                }).join("");
            }
        } catch (error) {
            console.error("Error loading employees list:", error);
            // hardcode some fallbacks if the api is acting up
            employeeSelectEl.innerHTML = `
                <option value="1" ${employeeId === 1 ? 'selected' : ''}>Ankur Bag (Engineering)</option>
                <option value="2" ${employeeId === 2 ? 'selected' : ''}>Sayan Sarkar (Engineering)</option>
            `;
        }

        // listen for dropdown changes so we can tell the backend to switch targets
        employeeSelectEl.addEventListener("change", async (e) => {
            const newEmployeeId = parseInt(e.target.value);
            if (isNaN(newEmployeeId)) return;
            
            try {
                const response = await fetch(`/api/active_employee?employee_id=${newEmployeeId}`, {
                    method: "POST"
                });
                if (response.ok) {
                    employeeId = newEmployeeId;
                    
                    // clear the chart so it doesn't look weird while we load the new person's data
                    if (timeSplitChart) {
                        timeSplitChart.data.datasets[0].data = [0, 0, 0];
                        timeSplitChart.update();
                    }
                    
                    // instantly pull the fresh numbers
                    pollLiveStatus();
                    pollDailyAnalytics();
                    pollHistoryLogs();
                }
            } catch (error) {
                console.error("Error switching active employee:", error);
            }
        });

        // kick off the first fetch now that we know who we're looking at
        pollLiveStatus();
        pollDailyAnalytics();
        pollHistoryLogs();

        // set up timers to keep fetching data on repeat
        setInterval(pollLiveStatus, 1000);     // Every 1 second for live counters/timer
        setInterval(pollDailyAnalytics, 3000); // Every 3 seconds for doughnut chart
        setInterval(pollHistoryLogs, 5000);    // Every 5 seconds for timeline and table
    }

    initEmployeeSelector();

    // handy little function to turn seconds into a nice clock format
    function formatSecondsToHHMMSS(totalSeconds) {
        if (totalSeconds < 0 || isNaN(totalSeconds)) totalSeconds = 0;
        const hrs = Math.floor(totalSeconds / 3600);
        const mins = Math.floor((totalSeconds % 3600) / 60);
        const secs = totalSeconds % 60;
        return [
            hrs.toString().padStart(2, '0'),
            mins.toString().padStart(2, '0'),
            secs.toString().padStart(2, '0')
        ].join(':');
    }

    // turns an ugly iso date string into just the time part
    function formatTime(isoString) {
        if (!isoString) return '--:--:--';
        const date = new Date(isoString);
        if (isNaN(date.getTime())) return '--:--:--';
        return date.toTimeString().split(' ')[0];
    }

    // turns an ugly iso date string into a full readable date and time
    function formatDateTime(isoString) {
        if (!isoString) return '--:--:--';
        const date = new Date(isoString);
        if (isNaN(date.getTime())) return '--:--:--';
        const yyyy = date.getFullYear();
        const mm = String(date.getMonth() + 1).padStart(2, '0');
        const dd = String(date.getDate()).padStart(2, '0');
        const time = date.toTimeString().split(' ')[0];
        return `${yyyy}-${mm}-${dd} ${time}`;
    }

    // ----------------------------------------------------
    // getting our chart.js doughnut all set up
    // ----------------------------------------------------
    function initChart() {
        const ctx = document.getElementById("timeSplitChart").getContext("2d");
        
        timeSplitChart = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: ["Working", "Idle", "Absent"],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ["#16a34a", "#f59e0b", "#ef4444"],
                    borderWidth: 1,
                    borderColor: "#1e293b", // Matches panel color
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false // turning off the default legend because we built a prettier custom one
                    },
                    tooltip: {
                        backgroundColor: "#0f172a",
                        titleColor: "#cbd5e1",
                        bodyColor: "#f8fafc",
                        borderColor: "#334155",
                        borderWidth: 1,
                        padding: 8,
                        boxPadding: 4,
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const seconds = context.raw || 0;
                                return `${label}: ${formatSecondsToHHMMSS(seconds)}`;
                            }
                        }
                    }
                },
                cutout: "75%"
            }
        });
    }

    // ----------------------------------------------------
    // api fetching stuff - this is where we actually get our data
    // ----------------------------------------------------

    // 1. fetch the live stuff for the top banner and summary box
    async function pollLiveStatus() {
        try {
            const response = await fetch(`/api/activity/live?employee_id=${employeeId}`);
            if (!response.ok) return;
            const data = await response.json();

            // update the big status text and change its color
            metricStatusEl.textContent = data.status;
            blockStatusEl.className = "metric-block";
            
            if (data.status === "WORKING") {
                blockStatusEl.classList.add("state-working");
            } else if (data.status === "IDLE") {
                blockStatusEl.classList.add("state-idle");
            } else {
                blockStatusEl.classList.add("state-absent");
            }

            // slap the numbers into the top row
            metricWorkingEl.textContent = data.working_time;
            metricIdleEl.textContent = data.idle_time;
            metricAbsentEl.textContent = data.absent_time;
            metricProductivityEl.textContent = `${data.productivity_score_today.toFixed(1)}%`;

            // update the overlays on top of the video feed
            feedModeEl.textContent = data.is_mock ? "SIMULATING FEED" : "WEBCAM ACTIVE";
            feedConfidenceEl.textContent = `CONF: ${Math.round(data.confidence * 100)}%`;

            // fill out the shift summary panel
            summaryFirstEl.textContent = data.first_activity || '--:--:--';
            summaryLastEl.textContent = data.last_activity || '--:--:--';
            summaryMonitoredEl.textContent = data.total_monitored_time;
            summaryProdPctEl.textContent = `${data.productivity_score_today.toFixed(1)}%`;

            // dump the dev variables into the bottom panel
            if (debugRawScoreEl) debugRawScoreEl.textContent = (data.raw_score !== undefined) ? data.raw_score.toFixed(4) : "0.0000";
            if (debugSmoothedScoreEl) debugSmoothedScoreEl.textContent = (data.smoothed_score !== undefined) ? data.smoothed_score.toFixed(4) : "0.0000";
            if (debugThresholdEl) debugThresholdEl.textContent = (data.movement_threshold !== undefined) ? data.movement_threshold.toFixed(4) : "0.5000";
            if (debugIdleCdEl) {
                debugIdleCdEl.textContent = data.status === "WORKING" ? `${data.idle_countdown}s` : "--";
            }
            if (debugEpsilonEl) debugEpsilonEl.textContent = (data.epsilon_filter !== undefined) ? data.epsilon_filter.toFixed(4) : "0.0015";

            // everything is good, make the connection dot green
            document.querySelector(".connection-status").innerHTML = `
                <span class="status-dot green"></span>
                <span>DB ONLINE</span>
            `;

        } catch (error) {
            console.error("Error polling live status:", error);
            // uh oh, connection failed, make it red
            document.querySelector(".connection-status").innerHTML = `
                <span class="status-dot" style="background-color: #ef4444; box-shadow: 0 0 4px #ef4444;"></span>
                <span>CONN OFFLINE</span>
            `;
        }
    }

    // 2. fetch the daily numbers specifically for the doughnut chart
    async function pollDailyAnalytics() {
        try {
            const response = await fetch(`/api/analytics/daily?employee_id=${employeeId}`);
            if (!response.ok) return;
            const data = await response.json();

            if (timeSplitChart) {
                timeSplitChart.data.datasets[0].data = [
                    data.working_seconds,
                    data.idle_seconds,
                    data.absent_seconds
                ];
                timeSplitChart.update();
            }
        } catch (error) {
            console.error("Error polling daily analytics:", error);
        }
    }

    // 3. fetch the session logs for the timeline and table
    async function pollHistoryLogs() {
        try {
            const response = await fetch(`/api/activity/history?employee_id=${employeeId}&limit=15`);
            if (!response.ok) return;
            const logs = await response.json();

            // a. build the cool little timeline cards
            if (logs.length === 0) {
                timelineListEl.innerHTML = `
                    <div class="list-empty">
                        No logged session segments for today.
                    </div>
                `;
            } else {
                timelineListEl.innerHTML = logs.map(log => {
                    const startStr = formatTime(log.start_time);
                    const endStr = log.end_time ? formatTime(log.end_time) : "ACTIVE";
                    const stateLower = log.state.toLowerCase();
                    return `
                        <div class="timeline-segment state-${stateLower}">
                            <div class="seg-meta">
                                <span class="seg-title"><span class="dot"></span> ${log.state}</span>
                                <span class="seg-time">${startStr} - ${endStr}</span>
                            </div>
                            <div class="seg-stats">
                                <span class="seg-dur mono">${log.duration_formatted}</span>
                                <span class="seg-conf">Conf: ${Math.round(log.confidence * 100)}%</span>
                            </div>
                        </div>
                    `;
                }).join('');
            }

            // b. build the detailed rows for the table at the bottom
            if (logs.length === 0) {
                logsTableBodyEl.innerHTML = `
                    <tr>
                        <td colspan="6" class="table-empty">No database records found.</td>
                    </tr>
                `;
            } else {
                logsTableBodyEl.innerHTML = logs.map(log => {
                    const startFullStr = formatDateTime(log.start_time);
                    const endFullStr = log.end_time ? formatDateTime(log.end_time) : "ACTIVE";
                    const stateLower = log.state.toLowerCase();
                    const confPercent = Math.round(log.confidence * 100);
                    const notesDisplay = log.transition_reason
                        ? `${log.transition_reason}${log.notes ? ` | ${log.notes}` : ''}`
                        : (log.notes || '--');
                    return `
                        <tr>
                            <td class="mono">${startFullStr}</td>
                            <td class="mono">${endFullStr}</td>
                            <td><span class="status-pill ${stateLower}">${log.state}</span></td>
                            <td class="mono">${log.duration_formatted}</td>
                            <td class="mono">${confPercent}%</td>
                            <td>${notesDisplay}</td>
                        </tr>
                    `;
                }).join('');
            }

        } catch (error) {
            console.error("Error polling history logs:", error);
        }
    }
});
