// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Bar Chart Example
var ctx = document.getElementById("unemployment_months");
var unemployment_months = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: ["February", "March", "April", "May", "June"],
    datasets: [
        {
            label: "Leisure and Hospitality",
            backgroundColor: "rgb(204,37,41)",
            data: [5.7,8.1,39.3,35.9,28.9]
        },
        {
            label: "Retail Trade",
            backgroundColor: "rgb(132,186,91)",
            data: [3.5,5.3,18.6,16.6,12.0]
        },
        {
            label: "Education and Health Services",
            backgroundColor: "rgb(218,124,48)",
            data: [2.4,3.2,10.9,10.1,8.6]
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
          max: 40,
          maxTicksLimit: 5
        },
        gridLines: {
          display: true
        }
      }],
    },
    legend: {
      display: true
    }
  }
});
