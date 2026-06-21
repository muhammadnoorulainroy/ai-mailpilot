/**
 * Sidebar tab switching. Wires each tab control to toggle the active state and
 * show only the matching tab panel.
 */
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('tab--active'));
    tab.classList.add('tab--active');

    const tabName = (tab as HTMLElement).dataset.tab;
    document.querySelectorAll('.tab-panel').forEach((panel) => {
      (panel as HTMLElement).style.display = 'none';
    });

    const activePanel = document.getElementById(`${tabName}-tab`);
    if (activePanel) activePanel.style.display = 'block';
  });
});
