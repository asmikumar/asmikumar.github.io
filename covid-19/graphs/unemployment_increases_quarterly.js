// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Bar Chart Example
var ctx = document.getElementById("unemployment_increases_quarterly");
var unemployment_increases_quarterly = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: ["White", "Black", "Hispanic", "Men", "Women"],
    datasets: [
        {
            backgroundColor: ["rgba(0,153,255,1)", "rgba(102,102,204,1)", "rgba(255,204,0,1)", "rgba(102,204,0,1)", "rgba(255,51,0,1)"],
            data: [281.3,191.1,304.8,245.7,302.9]
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
