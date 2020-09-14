// Call the dataTables jQuery plugin
$(document).ready(function() {
  $('#percent_increase').DataTable({
	"order": false,
    "searching": false,   // Search Box will Be Disabled
	"info": false,         // Will show "1 to n of n entries" Text at bottom
	"lengthChange": false, // Show entries feature is disabled
	"pageLength": 4,
	"paging": false,
	"bSort" : false
	// .sorting, .sorting_asc, .sorting_desc {
 //    	background : none;
	// }
  });
});
