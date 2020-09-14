// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Bar Chart Example
var ctx = document.getElementById("wages_gender");
var wages_gender = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: ["Men", "Women"],
    datasets: [
        {
            backgroundColor: ["rgba(102,204,0,1)", "rgba(255,51,0,1)"],
            data: [21.00, 17.84]
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
