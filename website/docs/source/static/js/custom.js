/*
 * This script modifies external links in the Sphinx-generated HTML.
 * All external links (http/https) that are NOT marked as internal
 * will be opened in a new browser tab.
 *
 * We use jQuery's $(document).ready() to wait until the DOM is fully loaded.
 */
$(document).ready(function () {
    $('a[href^="http://"], a[href^="https://"]')
        .not('a[class*=internal]')
        .attr('target', '_blank');
});
