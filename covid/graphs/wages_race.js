// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Bar Chart Example
var ctx = document.getElementById("wages_race");
var wages_race = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: ["White", "Black", "Hispanic"],
    datasets: [
        {
            backgroundColor: ["rgba(0,153,255,1)", "rgba(102,102,204,1)", "rgba(255,204,0,1)"],
            data: [21.32, 16.12, 15.89]
        }
    ],
  },
  options: {
    maintainAspectRatio: false,
    scales: {
      xAxes: [{
        time: {
          unit: 'month'
        },
        gridLines: {
          display: false
        },
        ticks: {
          maxTicksLimit: 6
        }
      }],
      yAxes: [{
        ticks: {
          min: 0,
          maxTicksLimit: 20
        },
        gridLines: {
          display: true
        }
      }],
    },
    legend: {
      display: false
    }
  }
});
