// Call the dataTables jQuery plugin
$(document).ready(function() {
  $('#industry_race').DataTable({
	"order": [[ 1, "asc" ]],
	"info": false,
	"searching": false,   // Search Box will Be Disabled
	"lengthChange": false,
	"paging": false
  });
});
