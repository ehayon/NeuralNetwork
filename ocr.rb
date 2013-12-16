require './NeuralNetwork'

net = NeuralNetwork.new(256, 40, 10, 0.1, 0.8)

matrix = IO.readlines('numbers.data').map { |line| line.split.map(&:to_i) }
training_data = []

matrix.each do |row|
  training_data << [row[0..255], row[-10..-1]]
end

File.open('training_results.txt', 'w') do |file|
  File.open('training_epochs.txt', 'w') do |file_epochs|
    20.times do |iter|
      pp net.train(training_data.sample(60), iter, file, file_epochs)
    end
  end
end


net.test(training_data.sample(500))

