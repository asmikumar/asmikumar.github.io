// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Pie Chart Example
var ctx = document.getElementById("financial_activities_gender").getContext('2d');
var financial_activities_gender = new Chart(ctx, {
  type: 'pie',
  data: {
    labels: ["Men", "Women"],
    datasets: [{
      backgroundColor: ["rgba(102,204,0,1)", "rgba(255,51,0,1)"],
      data: [47.4, 52.6]
    }]
  },
  options: {
    maintainAspectRatio: false
  }
});